# -*- coding: utf-8 -*-
"""
Spyder Editor

Contains functions to create segmentation map and calculate shape asymmetry from that segementation map.
Also included are the galaxy cutout function and source masking class.

Created: January 2022 - Nick DeMarchi
"""

import numpy as np
import skimage.transform
from photutils import CircularAperture
from photutils.segmentation import detect_threshold, detect_sources
from scipy.ndimage import uniform_filter
import astropy.units as u
from astropy.coordinates import Angle, SkyCoord 
from astroquery.vizier import Vizier
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel
from astropy.io import fits
import warnings



def create_segmentation(data, nsigma=3.0, sigma_clip=3.0 , npixels=8, filter=False):
    
    '''

    Parameters
    ----------
    data : (NXM array)
           The galaxy image.
    nsigma : (float) optional
             The number of standard deviations per pixel above the background for which to consider a 
             pixel as possibly being part of a source. The default is 3.0.
    sigma_clip : (float) or (astropy.stats.SigmaClip), optional
                 A SigmaClip object that defines the sigma clipping parameters. The default is 3.0.
    npixels : (int), optional
              The minimum number of connected pixels, each greater than threshold, that an object must 
              have to be deblended. npixels must be a positive integer. The default is 8.
    filter : (bool), optional
             Whether or not we use a uniform filter on the segmentation image. The default is False.

    Raises
    ------
    ValueError
        Raises error if our data is not two dimensional.

    Returns
    -------
    (NXM array)
        The segmentation image which will have the same shape as data.

    '''
    
    if data.sndim != 2:
        
        raise ValueError('Data must be a two dimensional array')
        
    threshold = detect_threshold(data, nsigma=nsigma , sigma_clip=sigma_clip)
    
    segm = detect_sources(data, threshold=threshold, npixels=npixels)
    
    label = np.argmax(segm.areas) + 1

    source_segmap = 1*(segm == label)
    
    if filter is False:
        
        return source_segmap
    
    else:
       
       filtered_float = uniform_filter(np.float64(source_segmap), size=5)
    
       filtered_segmap = filtered_float > 0.5
       
       return filtered_segmap

    



def calc_shape_asymmetry(segmap, centre, rmax):
    '''

    Parameters
    ----------
    segmap : (NXM array of labels)
             An array consiting of the 'labels' 0...k for k object.
    centre : (x,y)
             The centre of our main target.
    rmax : (float)
            The user defined radius to rotate our image within.

    Returns
    -------
    As : (float)
         Calculated shape asymmetry following Pawlik et. al. 2016:
             As = sum(|I_180 - I|) / (2*sum(|I|))

    '''
    
    if (segmap, centre, rmax) is not None:

        # Now we rotate our segmentation map
        
        seg_180 = skimage.transform.rotate(segmap, 180.0, center=centre)
        
        if np.unique(seg_180).shape[0] > 2:
            
            print('Grabbing the largest segmentation')
            
            segmap_180 = 1*(seg_180 == seg_180.max())
            
        else:
            
            segmap_180 = seg_180

        
        ap = CircularAperture(centre, rmax)
        
        ap_sum = ap.do_photometry(np.abs(segmap), method='exact')[0][0]
        
        ap_diff = ap.do_photometry(np.abs(segmap-segmap_180), method='exact')[0][0]

        As = ap_diff / (2*ap_sum)
        
    else:
        
        raise ValueError('Need to import a segmentation map and a center and radius for rotation')

    return As


def galaxy_cutout(data, header, ra, dec, name, size=10.0, centre_and_radius=False):
   
    """
    Two-dimensional cutout for galaxy image.
    Parameters:
    -----------
    data : (N X M array)
          The galaxy image.
    header : (FITS file header)
            The fits file header for our image.
    ra : (float)
         The right ascension of our target galaxy.
    dec : (float)
          The declination of our target galaxy.
    name : (str)
           The galaxy name
    size : (float)
           The factor of R25 (optical size of our galaxy) we want the cutout to be. Default is 10.
    centre_and_radius : (bool)
                        If we want our function to return the galaxy centre and R25 radius.
                        Default is False.
    Returns:
    --------
         cutout : (N X M array)
                  The new cutout image
         centre : (2D array, optional)
                  The galaxy centre in pixels
         r : (float)
             The galaxy radius R25 in arc-seconds
    """
    # remove bad pixels from the data

    is_good = np.isfinite(data)

    data[~is_good] = 0

    # sky coordinate for galaxy centre in RA and DEC

    galaxy_position = SkyCoord(ra * u.deg, dec * u.deg, frame='icrs')

    # load WCS

    w = WCS(header)

    # load an optical radius, R25

    leda_query = Vizier.query_region(name, radius='0d0m20s', catalog='HyperLEDA')

    r = Angle((10 ** leda_query[0]['logD25'][0] * 0.1 / 60 / 2) * u.deg).arcsec

    cutout = Cutout2D(data, galaxy_position, size=size * r * u.arcsec, wcs=w)

    xp, yp = skycoord_to_pixel(galaxy_position, wcs=cutout.wcs)

    centre = (xp, yp)

    if not centre_and_radius:

        return cutout

    else:

        return cutout, centre, r
    


class MaskSources(object):

    def __init__(self, data, name, path_to_table, wcs = None, header = None, ps=None, include_galaxy=False, large_incl_corr=True, incl_limit=0.2):
        
        """
        
        Class for creating a mask for a galaxy image.
        
        Parameters
        ----------
        data : (N X M array)
               The galaxy image.
        
        wcs : (astropy.wcs.wcs.WCS)
              The World Coordinate System of our galaxy image
               
        header : optional (astropy.io.fits.header.Header)
                 If wcs is None then the FITS file header for our image must be provided.
                 
        name : (string)
               The galaxy name.
               
        path_to_table : (string)
                        This should be a path to a table that contains the columns 'RA', 'DEC', 'pa', 'inclination'
                        and 'Galaxy'.
                        NOTE: This was created for my purpose to use the VERTICO tables from Toby Brown 2021 paper. 
                        The __init__ can be edited to load these values itself rather than point to a directory 
                        where the table is stored.
                        
        ps : (float)
             The pixel scale four our image.
             
        include_galaxy : (bool)
                         If True we also mask our galaxy. Default is False.
                         
        large_incl_corr : (bool)
                          Large inclination correction. If True we correct for inclination based on parameter 'incl_limit'.
                          The default is True.
                          
        incl_limit : (float)
                     The cos(inclination) limit. Default is cos(inclination) = 0.2.
        Returns
        -------
                The boolean mask for our image; under ProfileMask.disk_mask
                
        """
        
        if data.ndim != 2:
            raise ValueError('data must be 2 dimensional image')
        if ps is None:
            raise ValueError('pixel-scale must be specified')
        
        if wcs is None and header is not None:
            self.w = WCS(header)
        elif wcs is not None:
            self.w = wcs
        else:
            raise ValueError('wcs and/or header were not provided. One must be provided')

        self.data = data
        self.name = name
        self.path_to_table = path_to_table
        self.ps = ps
        self.include_galaxy = include_galaxy
        self.large_incl_corr = large_incl_corr
        self.incl_limit = incl_limit
        # run our nested function
        self.disk_mask = self.calc_mask()
    
    
    def calc_mask(self):
        '''
        
        Main function that will calculate the boolean mask. Nested is a function that produces a radius map
        to select sources to mask. 
        Returns
        -------
        (bool)
            Boolean mask of sources to mask
        '''
        
        def radius_map(shape, ra, dec, pa, incl, w, large_incl_corr=True,incl_limit=0.2):
            '''
            
            A function to create a radius map. We can select specific sources and mask them
            to a certain size. We choose 1.5 times the optical size R25 for the mask.
            
            '''

            # All inputs assumed as Angle
            if large_incl_corr and (np.isnan(pa.rad + incl.rad)):
                pa = Angle(0 * u.rad)
                incl = Angle(0 * u.rad)
                # Not written to the header
                msg = 'PA or INCL is NaN in radius calculation, setting both to zero'
                warnings.warn(msg, UserWarning)
                # Warning ends
            cos_pa, sin_pa = np.cos(pa.rad), np.sin(pa.rad)
            cos_incl = np.cos(incl.rad)
            if large_incl_corr and (cos_incl < incl_limit):
                cos_incl = incl_limit
            xcm, ycm = ra.rad, dec.rad
            coordinates = np.zeros(list(shape) + [2])
            # Original coordinate is (y, x)
            # :1 --> x, RA --> the one needed to be divided by cos(incl)
            # :0 --> y, Dec
            coordinates[:, :, 0], coordinates[:, :, 1] = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
            # Now, value inside coordinates is (x, y)
            # :0 --> x, RA --> the one needed to be divided by cos(incl)
            # :1 --> y, Dec
            for i in range(shape[0]):
                coordinates[i] = Angle(w.wcs_pix2world(coordinates[i], 1) * u.deg).rad
            coordinates[:, :, 0] = 0.5 * (coordinates[:, :, 0] - xcm) * (np.cos(coordinates[:, :, 1]) + np.cos(ycm))
            coordinates[:, :, 1] -= ycm
            # Now, coordinates is (dx, dy) in the original coordinate
            # cos_pa*dy-sin_pa*dx is new y
            # cos_pa*dx+sin_pa*dy is new x
            radius = np.sqrt((cos_pa * coordinates[:, :, 1] + sin_pa * coordinates[:, :, 0]) ** 2 + (
                    (cos_pa * coordinates[:, :, 0] - sin_pa * coordinates[:, :, 1]) / cos_incl) ** 2)
            radius = Angle(radius * u.rad).arcsec
            return radius
        
        # initialize boolean mask
        disk_mask = np.ones(self.data.shape, dtype='int64') < 0
        # first step is to mask any bad pixels or edges
        initial_mask = ~np.isfinite(self.data)
        disk_mask |= initial_mask
        # next mask all other known galaxies in sample
        load_table = fits.open(self.path_to_table)[1].data
        if self.include_galaxy:
            table = load_table
        else:
            table = load_table[~np.isin(load_table['Galaxy'], self.name)]
        pos = table['RA'] * u.deg, table['DEC'] * u.deg, table['pa'] * u.deg, table['inclination'] * u.deg
        ra, dec, pa, incl = Angle(pos[0]), Angle(pos[1]), Angle(pos[2]), Angle(pos[3])
        table_length = len(table['Galaxy'])
        for i in range(table_length):
            query = Vizier.query_region(table['Galaxy'][i], radius='0d0m20s', catalog='HyperLEDA')
            r25_extra = Angle((10 ** query[0]['logD25'][0] * 0.1 / 60 / 2) * u.deg).arcsec
            radius_map_extra = radius_map(self.data.shape, ra[i], dec[i], pa[i], incl[i], self.w)
            disk_mask |= radius_map_extra < (1.5 * r25_extra)
        # now load our galaxy ra,dec
        n_load_table = fits.open(self.path_to_table)[1].data
        n_table = n_load_table[np.isin(n_load_table['Galaxy'], self.name)]
        pos = n_table['RA'] * u.deg, n_table['DEC'] * u.deg
        n_ra, n_dec = Angle(pos[0]), Angle(pos[1])
        # Mask out other objects in the field of view
        c0 = SkyCoord(n_ra, n_dec, frame='icrs', equinox="J2000")
        # choose field of view to be size of input data
        leda_query = Vizier.query_region(c0,
                                         width=self.ps * self.data.shape[0] * u.arcsec,
                                         height=self.ps * self.data.shape[1] * u.arcsec,
                                         catalog='HyperLEDA')
        n_objects = len(leda_query[0]['RAJ2000'])
        print('number of masked objects is: ', n_objects)
        for j in range(n_objects):
            # need to make sure my galaxy exists in field of view and is not masked
            if len(leda_query[0]['ANames'][j].split()) > 0:
                n_names = len(leda_query[0]['ANames'][j].split())
                for k in range(n_names):
                    if leda_query[0]['ANames'][j].split()[k] == self.name:
                        print(leda_query[0]['ANames'][j].split()[k])
                        print('found you: ', self.name)
                        continue
            else:
                gal_ra_parsed = leda_query[0]['RAJ2000'][j].split(' ')
                ra_j = Angle(gal_ra_parsed[0] + 'h' + gal_ra_parsed[1] + 'm' + gal_ra_parsed[2] + 's')
                gal_dec_parsed = leda_query[0]['DEJ2000'][j].split(' ')
                dec_j = Angle(gal_dec_parsed[0] + 'd' + gal_dec_parsed[1] + 'm' + gal_dec_parsed[2] + 's')
                # 'logR25' # Axis ratio in log scale
                incl_j = Angle((np.arccos(1. / (10 ** leda_query[0]['logR25'][j])) * 180. / np.pi) * u.deg)
                # 'PA' # Position angle
                pa_j = Angle(leda_query[0]['PA'][j] * u.deg)
                # 'logD25' # Apparent diameter
                r25_j = Angle((10 ** leda_query[0]['logD25'][j] * 0.1 / 60 / 2) * u.deg).arcsec
                # create radius map
                radius_map_j = radius_map(self.data.shape, ra_j, dec_j, pa_j, incl_j, self.w)
                # add elliptical mask for specified extra source
                disk_mask |= radius_map_j < (1.5 * r25_j)
        # the final return
        return (~disk_mask)


