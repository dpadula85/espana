#!/usr/bin/env python

import numpy as np
from gridData import OpenDX, Grid
from rdkit.Chem import GetPeriodicTable
from scipy.spatial.distance import cdist

from espana.cube import Cube
from espana.selections import *

au2ang = 0.5291771
ang2au = 1 / au2ang
au2eV = 27.21138505


def centroid(coords, masses=None):
    '''
    Function to compute the centre (or the centre of mass) of a set of
    coordinates.

    Parameters
    ----------
    coord: np.array (N,3).
        coordinates.
    masses: np.array (N) (default: None).
        masses.

    Returns
    -------
    com: np.array (3).
        centre (or centre of mass) of the set of coordinates.
    '''

    com = np.average(coords, axis=0, weights=masses)

    return com


def lstsq_fit(pts):
    '''
    Function to fit a set of points with least squares. The geometrical objects
    involved depend on the number of dimensions (columns) of pts.

    Parameters
    ----------
    pts: np.array (N,M).
        coordinates.

    Returns
    -------
    coeffs: np.array (M).
        coefficients of the least squares fit.
    '''

    A = np.c_[ pts[:,:-1], np.ones(pts.shape[0]) ]
    B = pts[:,-1]

    coeffs, res, rank, singular_values = np.linalg.lstsq(A, B, rcond=None)

    return coeffs


def make_grid(**kwargs):
    '''
    Function to create a grid of points centred at the origin according to
    the basis vectors. Optional kwargs can control both the reference frame
    and whether the grid will be generated in space, onto a plane, or along a
    line.

    Parameters
    ----------
    ref: np.array (3,3).
        basis vectors.
    origin: np.array (3).
        coordinated of the origin.
    xu, yu, zu: float.
        maximum coefficient for each basis vector.
    xl, yl, zl: float.
        minimum coefficient for each basis vector.
    nx, ny, nz: int.
        number of points along each direction.

    Returns
    -------
    grid: np.array (M,3).
        grid of points.
    '''

    # Assign default reference system and origin to the cartesian ref frame
    ref = kwargs.pop("ref", np.eye(3))
    origin = kwargs.pop("origin", np.zeros(3))

    # Define options for the grid
    xu = kwargs.pop("xu", 5)
    yu = kwargs.pop("yu", 5)
    zu = kwargs.pop("zu", 5)
    xl = kwargs.pop("xl", -xu)
    yl = kwargs.pop("yl", -yu)
    zl = kwargs.pop("zl", -zu)
    nx = kwargs.pop("nx", np.abs(xl) + np.abs(xu) + 1)
    ny = kwargs.pop("ny", np.abs(yl) + np.abs(yu) + 1)
    nz = kwargs.pop("nz", np.abs(zl) + np.abs(zu) + 1)

    # Define spacings along each basis vector
    i = np.linspace(xl, xu, nx)
    j = np.linspace(yl, yu, ny)
    k = np.linspace(zl, zu, nz)

    # We should do
    # for p in i:
    #     for q in j:
    #         for r in k:
    #             gridpoint = origin + p * e1 + q * e2 + r * e3
    #
    # where e1, e2, e3 are basis vectors stored as columns of ref

    # Here is a vectorised version of the nested for loop
    # Make grid of displacements along each basis vector
    g = np.meshgrid(i, j, k, indexing='ij')

    # Convert to a more natural format, one column for each basis vector
    grid = np.vstack(list(map(np.ravel, g))).T

    # Transform each grid point to local basis and translate
    grid = np.dot(ref, grid.T).T + origin

    return grid


def proj(pts, coeffs):
    '''
    Function to compute projections of a set of points onto a plane.

    Parameters
    ----------
    pts: np.array (N,M).
        set of points
    coeffs: np.array (M).
        normal vector describing the plane.

    Returns
    -------
    prjs: np.array (N,M).
        set of points projected onto the plane.
    '''

    # For each point p in pts we should do
    # prj = p - np.dot(p, u) * u
    # where u is the normal unit vector.

    # Compute normal unit vector
    u = coeffs / np.linalg.norm(coeffs)

    # Here is a vectorised version of the projection formula
    # Compute elementwise dot products between pts and u.
    # Deal with the case of a single point as an exception.
    try:
        dotprods = np.einsum('ij,ij->i', pts, u.reshape(1,-1))
    except:
        dotprods = np.einsum('ij,ij->i', pts.reshape(-1,u.shape[0]), u.reshape(1,-1))

    # Repeat dot products and unit vector for elementwise mult to be
    # subtracted from originary coordinates
    dotprodsmat = np.repeat(dotprods.reshape(-1,1), u.shape[0], axis=1)
    umat = np.repeat(u.reshape(1,-1), dotprods.shape[0], axis=0)

    # Subtract the components along the normal to the plane from the
    # originary coordinates
    prjs = pts - dotprodsmat * umat

    return prjs


def make_plane_basis(plane):
    '''
    Function to define a local reference frame on a plane defined by its
    normal vector.

    Parameters
    ----------
    plane: np.array (N).
        normal vector describing the plane.

    Returns
    -------
    ref: np.array (N,N).
        local basis.
    '''

    # Get missing coeff
    u = plane / np.linalg.norm(plane)
    p = np.zeros(3)
    p[-1] = plane[-1]
    d = np.dot(-p, u)

    # Define two points on the plane
    p1 = np.array([0, 0, d / u[2]])
    p2 = np.array([0, d / u[1], 0])
    e1 = p2 - p1
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(u, e1)

    # Define the local reference frame basis and check its righthandedness
    ref = np.c_[ e1, e2, u ]
    det = np.linalg.det(ref)
    if det < 0.0:
        ref[:,0] = -ref[:,0]

    return ref


def sphere_fib(r=1, n=20):
    '''
    Fibonacci golden spiral algorithm to
    evenly distribute points on a sphere.

    Parameters
    ----------
    r: int or float.
        radius of the reference sphere.
    n: int.
        number of points to pack on the sphere.

    Returns
    -------
    coords: np.array(N, 3).
        coordinates of points onto sphere.
    '''

    r = float(r)
    angle = np.pi * (3. - np.sqrt(5.))
    theta = np.arange(n) * angle
    step = r / n
    z = np.linspace(r - r / n , r / n - r, n)
    rr = np.sqrt(r**2 - z**2)

    coords = np.zeros((n,3))
    coords[:,0] = rr * np.cos(theta)
    coords[:,1] = rr * np.sin(theta)
    coords[:,2] = z

    return np.array(coords)


def vdwsurf(coords, radii=None, scale=1, density=1, tol=0.001):
    '''
    Function to create the VdW surface for a given set of points.
    Places unit spheres on nucleii if radii are not provided.

    Parameters
    ----------
    coords: np.array (N,3).
        points (atoms) coordinates.
    radii: list or np.array (N).
        Van der Waals radii associated to points.
    scale: float.
        factor scaling the vdw radii.
    density: int or float.
        density of points to pack onto the spheres.
    tol:
        tolerance parameter to ensure a point on its
        vdw sphere is not excluded from the same sphere.

    Returns
    -------
    surf: np.array (N, 3).
        coordinates of calculated vdw surface points.
    '''

    if radii is None:
        radii = np.ones(coords.shape[0])

    radii *= scale

    # Create spheres
    surf = []
    for i in range(len(coords)):
        n = int(np.ceil(density * 4 * np.pi* radii[i]**2))
        dots = sphere_fib(radii[i], n)
        dots = coords[i] + dots
        surf.extend(dots)

    surf = np.array(surf)

    # Compute distance between coordinates and surface points
    # Filter out all those points that fall within the radius
    # of any atom.
    D = cdist(coords, surf)
    m = D.shape[1]
    F = np.repeat(radii.reshape(-1, 1), m, axis=1)
    filtered = D >= F - tol
    X = np.all(filtered, axis=0)
    idxs = np.where(X == True)
    surf = surf[idxs[0]]

    return surf


def grid_gen(pts, **kwargs):
    '''
    Function to create a grid around pts.

    Parameters
    ----------
    pts: np.array (N,3).
        set of points to be enclosed in the grid.
    type: str.
        type of grid to compute, either cubic (cub), VdW surface (vdw), or
        their difference (hollow).
    masses: np.array (N).
        atomic masses associated to each coordinate in pts, used to compute
        the centre of mass instead of the geometric centre.
    radii: np.array (N).
        atomic (VdW) radii associated to each coordinate in pts, used to compute
        VdW surface grid or hollow grid.
    sel: np.array (P).
        selection of atoms to fit the best plane passing through them as basis
        plane for the generation of the grid.
    nx, ny, nz: int.
        number of points along each direction for the calculation of the grid.
    density: int.
        number of points to pack onto VdW spheres for VdW surface grid.

    Returns
    -------
    grid: np.array (M,3).
        grid of points.
    prjs: np.array (N,3).
        projections of pts onto their best fit plane (not for vdw type).
    '''

    gridtype = kwargs.pop('type', 'cub')
    selg = kwargs.pop('sel', None)
    selp = kwargs.pop('pi', None)

    if gridtype in [ 'cub', 'hollow' ]:
        masses = kwargs.pop('masses', None)

        # Compute com
        if selg is not None:
            com = centroid(pts[selg], masses[selg])
        else:
            com = centroid(pts, masses)

        # Fit a plane - Deal with an optional selection to be read from the kwargs
        if selp is not None:
            plane = lstsq_fit(pts[selp])
        elif selg is not None:
            plane = lstsq_fit(pts[selg])
        else:
            plane = lstsq_fit(pts)

        # Define plane unit vector
        plane = plane / np.linalg.norm(plane)

        # Get com projection on the plane
        prjcom = proj(com, plane)

        # Project all points on the plane
        if selg is not None:
            prjs = proj(pts[selg], plane)
            pts2 = pts[selg]
        else:
            prjs = proj(pts, plane)
            pts2 = pts

        # Define plane local basis
        ref = make_plane_basis(plane)

        # Find displacements of basis vectors enclosing prjs
        # This means to find the highest dot product between each of prjs and
        # each basis vector describing the plane. Add some tolerance so as not
        # to be too close to the boundaries of the grid
        dotsx = np.einsum('ij,ij->i', prjs, ref[:,0].reshape(1,-1))
        dotsy = np.einsum('ij,ij->i', prjs, ref[:,1].reshape(1,-1))
        dotsz = np.einsum('ij,ij->i', pts2, ref[:,2].reshape(1,-1))

        xl = dotsx.min()
        xl = int(np.sign(xl) * np.ceil(np.abs(xl)))
        xl += np.sign(xl)
        xu = dotsx.max()
        xu = int(np.sign(xu) * np.ceil(np.abs(xu)))
        xu += np.sign(xu)

        yl = dotsy.min()
        yl = int(np.sign(yl) * np.ceil(np.abs(yl)))
        yl += np.sign(yl)
        yu = dotsy.max()
        yu = int(np.sign(yu) * np.ceil(np.abs(yu)))
        yu += np.sign(yu)

        zl = dotsz.min()
        zl = int(np.sign(zl) * np.ceil(np.abs(zl)))
        zl += np.sign(zl)
        zu = dotsz.max()
        zu = int(np.sign(zu) * np.ceil(np.abs(zu)))
        zu += np.sign(zu)

        # Define grid options
        if kwargs['nx'] == 0:
            xu = 0
            xl = 0
            kwargs['nx'] = 1

        if kwargs['ny'] == 0:
            yu = 0
            yl = 0
            kwargs['ny'] = 1

        if kwargs['nz'] == 0:
            zu = 0
            zl = 0
            kwargs['nz'] = 1

        kwargs['origin'] = prjcom
        kwargs['ref'] = ref
        kwargs['xu'] = xu
        kwargs['xl'] = xl
        kwargs['yu'] = yu
        kwargs['yl'] = yl
        kwargs['zu'] = zu
        kwargs['zl'] = zl

        grid = make_grid(**kwargs)

        if gridtype == 'hollow':
            radii = kwargs.pop('radii', np.ones(pts.shape[0]))

            if selg is not None:
                D = cdist(pts[selg], grid)
            else:
                D = cdist(pts, grid)

            m = D.shape[1]
            F = np.repeat(radii.reshape(-1, 1), m, axis=1)
            filtered = D >= F
            X = np.all(filtered, axis=0)
            idxs = np.where(X == True)
            grid = grid[idxs[0]]

        ### # Bonus stuff
        ### # ref is also the transformation matrix for passing from cartesian to
        ### # local basis.
        ### # Rotate the coordinates of the grid to the cartesian basis,
        ### # i.e. left multiply by ref.T
        ### gridc = np.dot(ref.T, (grid - com).T).T

        ### # To do any transformation in the local reference,
        ### # sandwich a transformation matrix M expressed in cartesian basis
        ### # between ref and ref.T,
        ### # e.g. M' = ref.dot(M).dot(ref.T)
        ### # Transformation example. Rotate around the basis vector corresponding to
        ### # y by 90 deg.
        ### M = rot([0, 1, 0], 90)
        ### gridl_rot = ref.dot(M).dot(ref.T).dot((grid - com).T).T + com

    elif gridtype == 'vdw':
        den = kwargs.pop('density', 10)
        radii = kwargs.pop('radii', np.ones(pts.shape[0]))

        if selg is not None:
            grid = vdwsurf(pts[selg], radii=radii[selg], density=den)
        else:
            grid = vdwsurf(pts, radii=radii, density=den)

        prjs = None
        ref = np.eye(3)

    return grid, prjs, ref


def read_sel(string):
    '''
    Function to parse cli selection, whether it is in a file or given
    explicitly in the cli.

    Parameters
    ----------
    string: str.
        selection string.

    Returns
    -------
    sel: str.
        selection string.
    '''

    try:
        with open(string) as f:
            string = f.readlines()

        string = list(map(str.rstrip, string))

    except IOError:
        string = [ string ]

    return string


def main():
    '''
    Cli call function.
    '''

    import argparse as arg
    import MDAnalysis as mda

    # Define cli options
    parser = arg.ArgumentParser(description='''Makes a grid of points around
                                an input geometry.''',
                                formatter_class=arg.ArgumentDefaultsHelpFormatter)

    # Input options
    inp = parser.add_argument_group('Input Data')

    inp.add_argument('-i', '--inp', required=True, type=str, dest='InpFile',
                     help='''Geometry Input File.''')

    # Calculation options
    calc = parser.add_argument_group('Calculation Options')

    calc.add_argument('--pi', default='all', dest='pi', type=str,
                      help='''Selection of atoms to look for pi systems.''')

    calc.add_argument('--type', default='cub', type=str, dest='type',
                      choices=['cub', 'vdw', 'hollow'],
                      help='''Desired grid type.''')

    calc.add_argument('--nx', default=50, type=int, dest='nx',
                      help='''Number of points along the first basis
                      vector.''')

    calc.add_argument('--ny', default=50, type=int, dest='ny',
                      help='''Number of points along the second basis
                      vector.''')

    calc.add_argument('--nz', default=50, type=int, dest='nz',
                      help='''Number of points along the third basis
                      vector.''')

    calc.add_argument('-d', '--density', default=10, type=int, dest='density',
                      help='''Density of points on VdW spheres.''')

    # Output options
    out = parser.add_argument_group('Output Options')

    out.add_argument('-o', '--out', default='grid', type=str, dest='OutRoot',
                     help='''Output File.''')

    out.add_argument('--fmt', default=['xyz', 'cub'], type=str, nargs='+',
                     choices=[ 'xyz' , 'cub', 'dx' ], dest='OutFile',
                     help='''Output file format.''')


    # Parse options
    args = parser.parse_args()
    kwargs = vars(args)

    # Import periodic table for chemical data
    pt = GetPeriodicTable()

    # Read structure
    u = mda.Universe(kwargs['InpFile'])

    # Get data needed to do calculations
    pts = u.atoms.positions
    atoms = u.atoms.types
    masses = np.array([ pt.GetAtomicWeight(x) for x in atoms ])

    try:
        radii = u.atoms.radii
    except:
        radii = np.array([ pt.GetRvdw(x) for x in atoms ])

    kwargs['radii'] = radii
    kwargs['masses'] = masses

    # Get idxs of the pi systems
    if kwargs['pi']:
        sels = read_sel(kwargs['pi'])
        idxs = []
        for sel in sels:
            selidxs = get_pi_system(u.atoms.select_atoms(sel))
            idxs.extend(selidxs)

        idxs = np.array(idxs)
        kwargs['pi'] = idxs

    # Call function to do calculations
    grid, prjs, U = grid_gen(pts, **kwargs)

    # Transform data aligning them to the cartesian basis to deal with
    # annoying cube / dx format
    com = centroid(pts, masses)
    pts = np.dot(U.T, (pts - com).T).T + com
    grid = np.dot(U.T, (grid - com).T).T + com
    deltas = np.diag(np.amax(np.diff(grid, axis=0), axis=0))

    if 'xyz' in kwargs['OutFile']:
        outname = kwargs['OutRoot'] + '.xyz'
        with open(outname, 'w') as f:
            f.write('%d\n' % grid.shape[0])
            titles = [ 'Dummy atom', 'X', 'Y', 'Z' ]
            f.write('%10s ' * len(titles) % tuple(titles))
            f.write('\n')
            np.savetxt(f, np.c_[ np.ones(grid.shape[0]), grid ],
                       fmt='%-10d %10.6f %10.6f %10.6f')

    if 'cub' in kwargs['OutFile'] or 'dx' in kwargs['OutFile']:

        if kwargs['nx'] == 0:
            kwargs['nx'] = 1

        if kwargs['ny'] == 0:
            kwargs['ny'] = 1

        if kwargs['nz'] == 0:
            kwargs['nz'] = 1

        voldata = { 'atoms'  : np.array([ pt.GetAtomicNumber(x) for x in atoms ]),
                    'natoms' : atoms.shape[0],
                    'coords' : pts * ang2au,
                    'origin' : np.amin(grid, axis=0) * ang2au,
                    'X'      : deltas[0] * ang2au,
                    'Y'      : deltas[1] * ang2au,
                    'Z'      : deltas[2] * ang2au,
                    'NX'     : kwargs['nx'],
                    'NY'     : kwargs['ny'],
                    'NZ'     : kwargs['nz'],
                    'grid'   : grid * ang2au,
                    'data'   : np.zeros((kwargs['nx'], kwargs['ny'], kwargs['nz']))
                    }

        if 'cub' in kwargs['OutFile']:
            outname = kwargs['OutRoot'] + '.cub'
            cubobj = Cube(**voldata)
            cubobj.dump(outname)

        if 'dx' in kwargs['OutFile']:
            outname = kwargs['OutRoot'] + '.dx'
            shape = ( voldata['NX'], voldata['NY'], voldata['NZ'] )
            origin = voldata['origin'] * au2ang
            delta = deltas
            dx = OpenDX.field('density', components=dict(
                positions=OpenDX.gridpositions(1, shape, origin, delta),
                connections=OpenDX.gridconnections(2, shape),
                data=OpenDX.array(3, voldata['data'])
                ))
            dx.write(outname)

    return


if __name__ == '__main__':
    main()
