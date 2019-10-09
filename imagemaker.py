# -*- coding: utf-8 -*-
"""
Utility functions for simulating image of atomic lattice

@author: Maxim Ziatdinov

"""

import numpy as np
import scipy.spatial as spatial
from scipy import ndimage
import itertools
from elements import ELEMENTS

def znumber(atomstr):
    '''Returns a dictionary value of Z
       number for specific element'''
    return ELEMENTS[atomstr].number

def atom_ch(atom_list):
    '''Returns dictionary with channel value
        for each unique atomic element'''
    species = np.unique(atom_list)
    z_numbers = [znumber(s) for s in species]
    z_numbers, species = zip(*sorted(zip(z_numbers, species), reverse=False))
    atom_ch_d = {}
    for i, s in enumerate(species):
        atom_ch_d[s] = i
    return atom_ch_d

def find_nn(p1, coord, r = 20):
    '''Find nearest neighbors for atom within specified radius r'''
    lattice_coord = np.copy(coord)
    lattice_coord_ = [p1]
    tree = spatial.cKDTree(lattice_coord)
    indices = tree.query_ball_point(lattice_coord_, r)
    indices = np.unique(list(itertools.chain.from_iterable(indices)))
    return indices


class MakeAtom:
    """
    Creates an image of atom modelled as
    2D Gaussian and a corresponding mask
    """
    def __init__(self, sc,  atom_str, r_mask=None, cfp=2,
                 theta=0, offset=0):
        """
        Args:
            sc (float): scale parameter, which determines Gaussian width
            atom_str (string): atom chemical identity
            r_mask (float): radius of mask corresponding to atom
            cfp (float): parameter n in the power law Z**n, where Z is atomic
            number
            theta (float): parameter of 2D gaussian function
            offset (float): parameter of 2D gaussian function
        """
        if sc % 2 == 0:
            sc += 1
        self.xo, self.yo = sc/2, sc/2
        x = np.linspace(0, sc, sc)
        y = np.linspace(0, sc, sc)
        self.x, self.y = np.meshgrid(x, y)
        self.sigma_x, self.sigma_y = sc/4, sc/4
        self.theta = theta
        self.offset = offset
        self.intensity = znumber(atom_str)**cfp
        if r_mask is None:
            self.r_mask = 3
        else:
            self.r_mask = r_mask

    def circularmask(self, image, radius):
        '''Returns a mask with specified radius'''
        h, w = self.x.shape
        X, Y = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X-self.xo+0.5)**2 + (Y-self.yo+0.5)**2)
        mask = dist_from_center <= radius
        image[~mask] = 0
        return image
    
    def atom2dgaussian(self):
        '''Models atom as 2d Gaussian'''
        a = (np.cos(self.theta)**2)/(2*self.sigma_x**2) +\
            (np.sin(self.theta)**2)/(2*self.sigma_y**2)
        b = -(np.sin(2*self.theta))/(4*self.sigma_x**2) +\
             (np.sin(2*self.theta))/(4*self.sigma_y**2)
        c = (np.sin(self.theta)**2)/(2*self.sigma_x**2) +\
            (np.cos(self.theta)**2)/(2*self.sigma_y**2)
        g = self.offset + self.intensity*np.exp(
            -(a*((self.x-self.xo)**2) + 2*b*(self.x-self.xo)*(self.y-self.yo) +\
            c*((self.y-self.yo)**2)))
        return g
    
    def atommask(self):
        '''Creates a mask for specific type of atom'''
        mask = self.atom2dgaussian()
        r1 = int(self.xo - self.r_mask/2)
        r2 = int(self.yo + self.r_mask/2)
        mask[mask > 0] = 1
        mask[r1:r2, r1:r2] = 1
        mask = self.circularmask(mask, self.r_mask/2)
        mask = mask[np.min(np.where(mask == 1)[0]):
                    np.max(np.where(mask == 1)[0]+1),
                    np.min(np.where(mask == 1)[1]):
                    np.max(np.where(mask == 1)[1])+1]
        return mask

class SimulateLattice():
    """
    Simulates image of atomic lattice from a 'coordinate file'
    """
    def __init__(self, acoordinates, ang2px, ang, sc,
                 nnd=1.42, randomize=0, convprobe=1.5, n_holes=0, r_hole=6,
                 impurity1='Si', impurity2='Si', n_impurity1=0, n_impurity2=0):
        """
        Args:
            coordinates (ndarray): set of atomic coordinates with
            corresponding atomic labels
            ang2px (float): angstrom-to-pixels conversion coefficent
            ang (tuple): list of angles for coordinates rotation (min,max,step)
            nnd (float): nearest neighbor distance in the lattice
            randomize (float): randomization of atomic coordinates
            convprobe (float): width of additional gaussian blurring to simulate a
                               convolution with probe
            n_holes (int): number of holes to be introduced
            r_hole (int): maximum radius of a hole in the units of nn distance
            impurity1 (str): type 1 substitutional impurity (simple substitution)
            impurity2 (str): type 2 substitutional impurity (remove a pair of nn atoms
                             and place an impurity in the middle)
            n_impurity1 (int): number of type 1 substittional impurities
            n_impurity2 (int): number of type 2 substitutional impurities
        """
        self.atom_list = acoordinates[:, 0]
        self.ang2px = ang2px
        if isinstance(ang, tuple):
            self.ang = np.random.choice(np.arange(ang[0], ang[1], ang[2]))
        else:
            self.ang = ang
        self.sc = np.random.choice(sc) if isinstance(sc, list) else sc
        self.nn_d = nnd
        self.convprobe = convprobe
        self.n_holes = n_holes
        self.r_hole = r_hole
        self.impurity1 = impurity1
        self.impurity2 = impurity2
        self.n_impurity1 = n_impurity1
        self.n_impurity2 = n_impurity2
    
        coordinates_x = np.array([np.float(i) for i in acoordinates[:, 1]])
        coordinates_y = np.array([np.float(i) for i in acoordinates[:, 2]])
        coordinates_xy = np.concatenate(
            (coordinates_x[:, None], coordinates_y[:, None]), axis=1)
        r, c = coordinates_xy.shape
        self.coordinates_xy = coordinates_xy +\
                    np.random.uniform(-randomize, +randomize, (r, c))

    def get_image_size(self):
        '''Establish max size of the image to be filled with atoms'''
        l_shape1 = np.ptp(self.coordinates_xy[:, 0]) * self.ang2px
        l_shape2= np.ptp(self.coordinates_xy[:, 1]) * self.ang2px
        return int(l_shape1), int(l_shape2)

    def rot_coordinates(self):
        '''Rotates coordinates'''
        def rot(xy, angle):
            '''Rotates coordinates around the center origin'''
            org_center = np.array([np.mean(xy[:, 0]), np.mean(xy[:, 1])])
            org = xy - org_center
            a = np.deg2rad(angle)
            new = np.array([org[:, 0] * np.cos(a) + org[:, 1] * np.sin(a),
                            -org[:, 0] * np.sin(a) + org[:, 1] * np.cos(a) ])
            return new.T+org_center
        coordinates_r = rot(self.coordinates_xy, self.ang)
        coord_to_del = np.concatenate(
            (np.where(coordinates_r[:, 0] < np.amin(self.coordinates_xy[:, 0]))[0],
             np.where(coordinates_r[:, 0] > np.amax(self.coordinates_xy[:, 0]))[0],
             np.where(coordinates_r[:, 1] < np.amin(self.coordinates_xy[:, 1]))[0],
             np.where(coordinates_r[:, 1] > np.amax(self.coordinates_xy[:, 1]))[0]),
            axis=0)
        coord_to_del = np.unique(coord_to_del)
        coordinates_r = np.delete(coordinates_r, coord_to_del, axis=0)
        atom_list_r = np.delete(np.copy(self.atom_list), coord_to_del, axis=0)
        return atom_list_r, coordinates_r

    def make_holes(self, atom_list, coordinates, rmin=3):
        '''Introduces holes into atomic lattice'''
        eps1 = 20
        eps2 = 10
        rmin = rmin
        rmax = self.r_hole
        xmin = np.amin(coordinates[:, 0]) + eps1
        xmax = np.amax(coordinates[:, 0]) - eps1
        ymin = np.amin(coordinates[:, 1]) + eps1
        ymax = np.amax(coordinates[:, 1]) - eps1
        centers_used = np.empty((0,2))
        if self.n_holes == 0:
            return np.copy(atom_list), np.copy(coordinates)
        i = 0
        while i < self.n_holes:
            center_x = np.random.randint(xmin, xmax)
            center_y = np.random.randint(ymin, ymax)
            center = np.array([center_y, center_x])
            center = np.expand_dims(center, axis=0)
            radius = np.random.randint(rmin, rmax)
            centers_used = np.append(centers_used, center, axis=0)
            if centers_used.shape[0] > 1:
                nearby_centers = find_nn(center, centers_used, r=radius+eps2)
                if len(nearby_centers) > 1:
                    continue
            vac_nn = find_nn(center, coordinates, radius)
            coordinates = np.delete(coordinates, vac_nn, axis=0)
            atom_list = np.delete(atom_list, vac_nn, axis=0)
            i += 1
        return atom_list, coordinates

    def implant_impurity1(self, atom_list, coordinates):
        '''Implant a substitutional impurity'''
        i = 0
        while i < self.n_impurity1:
            r = np.random.randint(0, len(atom_list)-1)
            atom_list[r] = self.impurity1
            i += 1
        return atom_list, coordinates

    def implant_impurity2(self, atom_list, coordinates):
        '''
        Remove two nearest-neighbors lattice atoms
        and implant impurity in a middle
        '''
        nn_d = self.nn_d
        i = 0
        while i < self.n_impurity2:
            coordinates_ = np.copy(coordinates)
            # Find a pair of nearest neighbor atoms
            index1 = np.random.randint(len(coordinates))
            c = coordinates[index1].reshape(1, 2)
            d, index1 = spatial.cKDTree(coordinates_).query(c)[:2]
            coordinates_[index1] = [-10., -10.]
            d, index2 = spatial.cKDTree(coordinates_).query(coordinates[index1])[:2]
            xy = np.concatenate((coordinates[index1], coordinates[index2]), axis=0)
            d = np.sqrt((xy[0,0] - xy[1,0])**2 + (xy[0,1] - xy[1,1])**2)
            # Define impurity coordinates
            impurity_coord = (xy[0]+xy[1])/2
            if d > nn_d + 0.2*nn_d:
                continue
            # Remove host lattice atoms and add an impurity
            coordinates = np.delete(coordinates, np.array([index1, index2], dtype=int), axis=0)
            atom_list = np.delete(atom_list, np.array([index1, index2], dtype=int), axis=0)
            coordinates = np.append(coordinates, [impurity_coord], axis=0)
            atom_list = np.append(atom_list, [self.impurity2], axis=0)
            i += 1
        return atom_list, coordinates

    def make_image(self, r_mask=3):
        '''Simulates image'''
        eps1 = 100#self.sc*8
        eps2 = 10#self.sc*2
        atom_list_r, coordinates_r = self.rot_coordinates()
        atom_list_r, coordinates_r = self.make_holes(atom_list_r,
                                                     coordinates_r)
        atom_list_r, coordinates_r = self.implant_impurity1(atom_list_r,
                                                       coordinates_r)
        atom_list_r, coordinates_r = self.implant_impurity2(atom_list_r,
                                                            coordinates_r)
        atom_ch_all = atom_ch(atom_list_r)
        l_shape1, l_shape2 = self.get_image_size()
        lattice = np.zeros((l_shape1 + eps1, l_shape2 + eps1))
        lattice_mask = np.zeros((l_shape1 + eps1, l_shape2 + eps1,
                                 len(np.unique(atom_list_r))))
        for atom_idx, coord in zip(atom_list_r, coordinates_r):
            coord_px = coord*self.ang2px
            channel = atom_ch_all[atom_idx]
            x = int(np.around(coord_px[0])) + eps2
            y = int(np.around(coord_px[1])) + eps2
            cfp=np.random.randint(200, 205)/100
            a = MakeAtom(self.sc, atom_idx, r_mask, cfp=2)
            atom = a.atom2dgaussian()
            mask = a.atommask()
            r_a = atom.shape[0]/2
            r_m = mask.shape[0]/2
            r_a1 = int(r_a - 0.5)
            r_a2 = int(r_a + 0.5)
            r_m1 = int(r_m - 0.5)
            r_m2 = int(r_m + 0.5)
            #print(lattice.shape, x, y, r_a1, r_a2)
            lattice[x-r_a1:x+r_a2, y-r_a1:y+r_a2] = atom
            lattice_mask[x-r_m1:x+r_m2, y-r_m1:y+r_m2, channel] = mask
        lattice = ndimage.filters.gaussian_filter(lattice, sigma=self.convprobe)
        return lattice, lattice_mask
