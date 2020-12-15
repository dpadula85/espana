#!/usr/bin/env python

import numpy as np
import MDAnalysis as mda


def get_pi_system(sel):
    '''
    Function to identify pi systems in a selection.

    Parameters
    ----------
    sel: object.
        MDAnalysis selection.

    Returns
    -------
    idxs: np.array (N).
        indices of atoms belonging to pi systems.
    '''

    # Make connectivity matrix
    sel.guess_bonds()
    bds = sel.bonds.to_indices()

    # Use -1 as placeholder for non-existing bonds
    conn = np.ones((len(sel.atoms), 4)) * -1
    for i, bond in enumerate(bds):
        at1, at2 = bond
        for j in np.arange(conn[i].shape[0]):
            if conn[i,j] == -1:
                conn[i,j] = at2
                break

        for j in np.arange(conn[i].shape[0]):
            if conn[i,j] == -1:
                conn[i,j] = at1
                break

    # Get heavy atoms
    atoms = sel.atoms.types
    heavy = atoms != "H"

    # Get atoms with connectivity < 4
    unsat = conn[:,-1] == -1

    # Get heavy with connectivity < 4
    idxs = np.where(heavy & unsat)
    idxs = sel.indices[idxs]

    return idxs


if __name__ == '__main__':
    pass
