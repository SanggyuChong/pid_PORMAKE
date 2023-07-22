import itertools
import ase
import ase.visualize
import numpy as np
import scipy

from .log import logger


def find_best_permutation(p, q):
    dist = p[:, np.newaxis] - q[np.newaxis, :]
    dist = np.linalg.norm(dist, axis=-1)

    _, perm = scipy.optimize.linear_sum_assignment(dist)

    return perm


def find_best_orientation(p, q):
    # This function gives root "sum" squared distance...
    U, rmsd = scipy.spatial.transform.Rotation.align_vectors(p, q)
    return U.as_matrix().T, np.sqrt(np.square(rmsd)/len(p))


def get_rotation_matrix_around_zvec(p, q):

    # assume alignment and drop the z component, then normalize
    p_xy = np.delete(p, np.s_[-1])
    q_xy = np.delete(q, np.s_[-1])
    p_xy = p_xy / np.linalg.norm(p_xy)
    q_xy = q_xy / np.linalg.norm(q_xy)

    costheta = np.clip(np.dot(p_xy, q_xy), -1, 1)
    sintheta = np.sin(np.arccos(costheta))

    Urot = [[costheta, -sintheta,  0],
            [sintheta,  costheta,  0],
            [       0,         0,  1]]
 
    angle = np.degrees(np.arccos(costheta))
    return np.array(Urot), angle

class Locator:
    def locate(self, target, bb, max_n_slices=4):
        """
        Locate building block (bb) to target_points
        using the connection points of the bb.

        Return:
            located building block and RMS.
        """
        local0 = target
        local1 = bb.local_structure()

        # p: target points, q: to be rotated points.
        p_coord = local0.atoms.positions

        # Serching best orientation over Euler angles.
        n_points = p_coord.shape[0]
        if n_points == 2:
            n_slices = 1
        elif n_points == 3:
            n_slices = max_n_slices - 2
        elif n_points == 4:
            n_slices = max_n_slices - 1
        else:
            n_slices = max_n_slices

        logger.debug("n_slices: %d", n_slices)

        alpha = np.linspace(0, 360, n_slices)
        beta = np.linspace(0, 180, n_slices)
        gamma = np.linspace(0, 360, n_slices)

        min_rmsd_val = 1e30
        for a, b, g in itertools.product(alpha, beta, gamma):
            # Copy atoms object for euler rotation.
            atoms = local1.atoms.copy()
            # Rotate.
            atoms.euler_rotate(a, b, g, center=(0, 0, 0))

            # Reorder coordinates.
            q_coord = atoms.positions
            q_perm = find_best_permutation(p_coord, q_coord)

            # Use this permutation of the euler angle. But do not used the
            # Rotated atoms in order to get pure U.
            q_coord = local1.atoms.positions[q_perm]

            # Rotation matrix.
            U, rmsd_val = find_best_orientation(p_coord, q_coord)

            # Save best U and Euler angle.
            if rmsd_val < min_rmsd_val:
                min_rmsd_val = rmsd_val
                min_rmsd_U = U
                min_perm = q_perm

            # The value of 1e-4 can be changed.
            if min_rmsd_val < 1e-4:
                break

        # Load best vals.
        U = min_rmsd_U
        rmsd_val = min_rmsd_val

        # Copy for ratation.
        bb = bb.copy()

        # Rotate using U from RMSD.
        positions = bb.atoms.positions
        centroid = bb.centroid

        positions -= centroid
        positions = np.dot(positions, U) + centroid

        # Update position of atoms.
        bb.atoms.set_positions(positions)

        return bb, min_perm, rmsd_val

    def locate_with_permutation(self, target, bb, permutation):
        """
        Locate bb to target with pre-obtained permutation of bb.
        """
        local0 = target
        local1 = bb.local_structure()

        # p: target points, q: to be rotated points.
        p_atoms = np.array(local0.atoms.symbols)
        p_coord = local0.atoms.positions

        q_atoms = np.array(local1.atoms.symbols)
        q_coord = local1.atoms.positions
        # Permutation used here.
        q_coord = q_coord[permutation]

        # Rotation matrix.
        U, rmsd_val = find_best_orientation(p_coord, q_coord)

        bb = bb.copy()

        # Rotate using U from RMSD.
        positions = bb.atoms.positions
        centroid = bb.centroid

        positions -= centroid
        positions = np.dot(positions, U) + centroid

        # Update position of atoms.
        bb.atoms.set_positions(positions)

        return bb, rmsd_val



    def locate_with_permutation_and_planarity_enforcement(self, target_with_scps, bb, permutation):
        """
        Locate bb to target with pre-obtained permutation of bb.
        This method, specialized for edges, additionally performs rotation of the edge
        along the edge vector to best match planarity conditions
        """
        local0 = target_with_scps
        local1 = bb.local_structure(scps=True)

        # p: target points, q: to be rotated points.
        p_atoms = np.array(local0.atoms.symbols)
        p_coord = local0.atoms.positions

        q_atoms = np.array(local1.atoms.symbols)
        q_coord = local1.atoms.positions
        # Permutation used here.
        q_coord = q_coord[permutation]

        # Rotation matrix.
        U, rmsd_val = find_best_orientation(p_coord, q_coord)

        bb = bb.copy()

        # Rotate using U from RMSD.
        positions = bb.atoms.positions
        centroid = bb.centroid

        positions -= centroid
        positions = np.dot(positions, U) + centroid

        # Update position of atoms.
        bb.atoms.set_positions(positions)

        #################################################################
        #                                                               #
        #  MODIFIED BY SRC TO PERFORM PLANARITY ENFORCEMENT (21/10/19)  #
        #                                                               #
        #################################################################

        # For the edge, update local structure
        local1 = bb.local_structure(scps=True)
        ## align coords and scps of local0 and local1 to z_vec to eliminate "connection" axis
        t_coord = local0.atoms.positions 
        e_coord = local0.atoms.positions
        t_scps = local0.scps
        e_scps = local1.scps

        zvec = np.array([[0, 0, -1],[0, 0, 1]])

        Uz, _ = find_best_orientation(zvec, t_coord) 

        t_coord = np.dot(t_coord, Uz)
        t_scps_aligned = np.dot(t_scps, Uz)
        e_coord = np.dot(e_coord, Uz)
        e_scps_aligned = np.dot(e_scps, Uz) 

        #view_int = ase.Atoms('HeC2N2O2F2', positions = np.concatenate([[[0, 0, 0]], t_coord, e_coord, t_scps_aligned, e_scps_aligned], axis=0))
        #ase.visualize.view(view_int)

        # drop the z component now that vectors have been aligned along z direction
        t_scps_xy = np.delete(t_scps_aligned, np.s_[-1], 1)
        e_scps_xy = np.delete(e_scps_aligned, np.s_[-1], 1)
        t_scps_n = t_scps_xy / np.linalg.norm(t_scps_xy, axis=1)
        e_scps_n = e_scps_xy / np.linalg.norm(e_scps_xy, axis=1)

        # three-dimensionalize
        t_scp1 = np.append(t_scps_n[0], 0)
        t_scp2 = np.append(t_scps_n[1], 0)
        e_scp1 = np.append(e_scps_n[0], 0)      
        e_scp2 = np.append(e_scps_n[1], 0)

        # obtain rotation matrix based on the 1st scp
        Urot, rot_angle = get_rotation_matrix_around_zvec(t_scps_aligned[0], e_scps_aligned[0]) 

        ## view rotated scps for debugging
        #e_scps_aligned = np.dot(e_scps_aligned, Urot)
        #view_int = ase.Atoms('HeC2O2F2', positions = np.concatenate([[[0, 0, 0]], t_coord, t_scps_aligned, e_scps_aligned], axis=0))
        #ase.visualize.view(view_int)

        # perform rotation to choose rotation direction and save residual angle
        e_scp1_rotated1 = np.dot(e_scp1, Urot)
        e_scp1_rotated2 = np.dot(e_scp1, Urot.T)
        e_scp2_rotated1 = np.dot(e_scp2, Urot)
        e_scp2_rotated2 = np.dot(e_scp2, Urot.T)

        check_angle1 = np.degrees(np.arccos(np.clip(np.dot(t_scp1, e_scp1_rotated1), -1, 1)))
        check_angle2 = np.degrees(np.arccos(np.clip(np.dot(t_scp1, e_scp1_rotated2), -1, 1)))
        resid_angle1 = np.degrees(np.arccos(np.clip(np.dot(t_scp2, e_scp2_rotated1), -1, 1)))
        resid_angle2 = np.degrees(np.arccos(np.clip(np.dot(t_scp2, e_scp2_rotated2), -1, 1)))
 
        if abs(check_angle1) > 90:
            check_angle1 = 180 - abs(check_angle1)
        if abs(check_angle2) > 90:
            check_angle2 = 180 - abs(check_angle2)
        if abs(resid_angle1) > 90:
            resid_angle1 = 180 - abs(resid_angle1)
        if abs(resid_angle2) > 90:
            resid_angle2 = 180 - abs(resid_angle2)

        if check_angle1 < check_angle2:
            resid_angle = resid_angle1
        else:
            resid_angle = resid_angle2
            Urot = Urot.T

        # we now align actual building block to z_vec
        positions = bb.atoms.positions
        centroid = bb.centroid
        positions -= centroid

        # alignment to z_vec
        positions = np.dot(positions, Uz)

        # rotation for planarity enforcement
        positions = np.dot(positions, Urot)

        # undo z_vec alignment and move it back
        positions = np.dot(positions, Uz.T) + centroid

        # save it to the building_block, pass it down to original code
        bb.atoms.set_positions(positions)

        return bb, rmsd_val, resid_angle

    def calculate_rmsd(self, target, bb, max_n_slices=6):
        _, _, rmsd_val = self.locate(target, bb, max_n_slices)
        return rmsd_val
