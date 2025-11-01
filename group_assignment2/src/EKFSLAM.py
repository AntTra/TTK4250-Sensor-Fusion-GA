from typing import Tuple
import numpy as np
from numpy import ndarray
from dataclasses import dataclass, field
from scipy.linalg import block_diag
import scipy.linalg as la
from utils import rotmat2d
from JCBB import JCBB
import utils
import solution.EKFSLAM


@dataclass
class EKFSLAM:
    Q: ndarray
    R: ndarray
    do_asso: bool
    alphas: 'ndarray[2]' = field(default=np.array([0.001, 0.0001]))
    sensor_offset: 'ndarray[2]' = field(default=np.zeros(2))

    def f(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Add the odometry u to the robot state x.

        Parameters
        ----------
        x : np.ndarray, shape=(3,)
            the robot state
        u : np.ndarray, shape=(3,)
            the odometry

        Returns
        -------
        np.ndarray, shape = (3,)
            the predicted state
        """
        # TODO replace this with your own code
        px, py, psi = x
        u_fwd, u_lat, omega = u
        c, s = np.cos(psi), np.sin(psi)
        xpred = np.array([
            px + u_fwd * c - u_lat * s,
            py + u_fwd * s + u_lat * c,
            utils.wrapToPi(psi + omega),
        ])
        return xpred

        # TODO, eq (11.7). Should wrap heading angle between (-pi, pi), see utils.wrapToPi
        xpred = None

        return xpred

    def Fx(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Calculate the Jacobian of f with respect to x.

        Parameters
        ----------
        x : np.ndarray, shape=(3,)
            the robot state
        u : np.ndarray, shape=(3,)
            the odometry

        Returns
        -------
        np.ndarray
            The Jacobian of f wrt. x.
        """
        # TODO replace this with your own code
        _, _, psi = x
        u_fwd, u_lat, _ = u
        c, s = np.cos(psi), np.sin(psi)
        Fx = np.array([
            [1.0, 0.0, -u_fwd * s - u_lat * c],
            [0.0, 1.0,  u_fwd * c - u_lat * s],
            [0.0, 0.0,  1.0],
        ])
        return Fx

        Fx = None  # TODO, eq (11.13)

        return Fx

    def Fu(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Calculate the Jacobian of f with respect to u.

        Parameters
        ----------
        x : np.ndarray, shape=(3,)
            the robot state
        u : np.ndarray, shape=(3,)
            the odometry

        Returns
        -------
        np.ndarray
            The Jacobian of f wrt. u.
        """
        # TODO replace this with your own code
        _, _, psi = x
        c, s = np.cos(psi), np.sin(psi)
        Fu = np.array([
            [ c, -s, 0.0],
            [ s,  c, 0.0],
            [0.0, 0.0, 1.0],
        ])
        return Fu

        Fu = None  # TODO, eq (11.14)

        return Fu

    def predict(
        self, eta: np.ndarray, P: np.ndarray, z_odo: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict the robot state using the zOdo as odometry the corresponding state&map covariance.

        Parameters
        ----------
        eta : np.ndarray, shape=(3 + 2*#landmarks,)
            the robot state and map concatenated
        P : np.ndarray, shape=(3 + 2*#landmarks,)*2
            the covariance of eta
        z_odo : np.ndarray, shape=(3,)
            the measured odometry

        Returns
        -------
        Tuple[np.ndarray, np.ndarray], shapes= (3 + 2*#landmarks,), (3 + 2*#landmarks,)*2
            predicted mean and covariance of eta.
        """
        # etapred, P = solution.EKFSLAM.EKFSLAM.predict(self, eta, P, z_odo)
        # return etapred, P

        # check inout matrix
        assert np.allclose(P, P.T), "EKFSLAM.predict: not symmetric P input"
        assert np.all(
            np.linalg.eigvals(P) >= 0
        ), "EKFSLAM.predict: non-positive eigen values in P input"
        assert (
            eta.shape * 2 == P.shape
        ), "EKFSLAM.predict: input eta and P shape do not match"
        etapred = np.empty_like(eta)

        x = eta[:3]
        etapred[:3] = self.f(x, z_odo)   # robot state prediction
        etapred[3:] = eta[3:]            # landmarks: no effect

        Fx = self.Fx(x, z_odo)           # TODO
        Fu = self.Fu(x, z_odo)           # TODO

        # evaluate covariance prediction in place to save computation
        # only robot state changes, so only rows and colums of robot state needs changing
        # cov matrix layout:
        # [[P_xx, P_xm],
        #  [P_mx, P_mm]]
        P[:3, :3] = Fx @ P[:3, :3] @ Fx.T + Fu @ self.Q @ Fu.T
        P[:3, 3:] = Fx @ P[:3, 3:]
        P[3:, :3] = P[:3, 3:].T

        assert np.allclose(P, P.T), "EKFSLAM.predict: not symmetric P"
        assert np.all(
            np.linalg.eigvals(P) > 0
        ), "EKFSLAM.predict: non-positive eigen values"
        assert (
            etapred.shape * 2 == P.shape
        ), "EKFSLAM.predict: calculated shapes does not match"

        return etapred, P

    def h(self, eta: np.ndarray) -> np.ndarray:
        """Predict all the landmark positions in sensor frame.

        Parameters
        ----------
        eta : np.ndarray, shape=(3 + 2 * #landmarks,)
            The robot state and landmarks stacked.

        Returns
        -------
        np.ndarray, shape=(2 * #landmarks,)
            The landmarks in the sensor frame.
        """

        # TODO replace this with your own code
        x = eta[0:3]
        # reshape map (2, #landmarks), m[:, j] is the jth landmark
        m = eta[3:].reshape((-1, 2)).T

        Rot = rotmat2d(-x[2])  # world -> sensor

        # relative position of landmark to sensor on robot in world frame
        sensor_pos_world = x[0:2] + rotmat2d(x[2]) @ self.sensor_offset
        delta_m = m - sensor_pos_world[:, None]

        # predicted measurements in cartesian coordinates in sensor frame
        zpredcart = Rot @ delta_m

        zpred_r = np.linalg.norm(zpredcart, axis=0)
        zpred_theta = np.arctan2(zpredcart[1, :], zpredcart[0, :])
        zpred = np.vstack((zpred_r, zpred_theta))

        # stack measurements along one dimension, [range1 bearing1 range2 bearing2 ...]
        zpred = zpred.T.ravel()

        assert (
            zpred.ndim == 1 and zpred.shape[0] == eta.shape[0] - 3
        ), "SLAM.h: Wrong shape on zpred"

        return zpred

    def h_jac(self, eta: np.ndarray) -> np.ndarray:
        """Calculate the jacobian of h.

        Parameters
        ----------
        eta : np.ndarray, shape=(3 + 2 * #landmarks,)
            The robot state and landmarks stacked.

        Returns
        -------
        np.ndarray, shape=(2 * #landmarks, 3 + 2 * #landmarks)
            the jacobian of h wrt. eta.
        """
        # extract states and map
        x = eta[0:3]
        # reshape map (2, #landmarks), m[j] is the jth landmark
        m = eta[3:].reshape((-1, 2)).T

        numM = m.shape[1]

        Rot = rotmat2d(x[2])          # R(psi) world->body inverse is Rot.T
        Rpihalf = rotmat2d(np.pi / 2)

        # relative position: m - (rho + R(psi)s_off)
        sensor_offset_world = Rot @ self.sensor_offset
        delta_m = m - (x[0:2][:, None] + sensor_offset_world[:, None])

        # world -> sensor frame
        zc = Rot.T @ delta_m

        zr = np.linalg.norm(zc, axis=0)
        zpred = np.vstack((zr, np.arctan2(zc[1, :], zc[0, :])))

        # Allocate H and set submatrices as memory views into H
        H = np.zeros((2 * numM, 3 + 2 * numM))
        Hx = H[:, :3]
        Hm = H[:, 3:]

        jac_z_cb = -np.eye(2, 3)
        for i in range(numM):  # loop is fine; can be vectorized later
            ind = 2 * i
            inds = slice(ind, ind + 2)

            zci = zc[:, i]
            dx, dy = zci[0], zci[1]
            r2 = dx * dx + dy * dy
            r = np.sqrt(r2)

            # polar Jacobian at zci
            Jpol = np.array([[dx / r,      dy / r],
                             [-dy / r2,     dx / r2]])

            # derivatives of zc wrt x,y and psi
            dzc_dxy = -Rot.T

            # d zc / d psi = -R(π/2)*zc - R(π/2)*s_off
            dzc_dpsi = -Rpihalf @ zci - Rpihalf @ self.sensor_offset

            Hx_i = np.zeros((2, 3))
            Hx_i[:, :2] = Jpol @ dzc_dxy
            Hx_i[:,  2] = Jpol @ dzc_dpsi

            # landmark block
            Hm_i = Jpol @ Rot.T

            Hx[inds, :] = Hx_i
            Hm[inds, 2 * i:2 * i + 2] = Hm_i

        return H

    def add_landmarks(
        self, eta: np.ndarray, P: np.ndarray, z: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate new landmarks, their covariances and add them to the state.

        Parameters
        ----------
        eta : np.ndarray, shape=(3 + 2*#landmarks,)
            the robot state and map concatenated
        P : np.ndarray, shape=(3 + 2*#landmarks,)*2
            the covariance of eta
        z : np.ndarray, shape(2 * #newlandmarks,)
            A set of measurements to create landmarks for

        Returns
        -------
        Tuple[np.ndarray, np.ndarray], shapes=(3 + 2*(#landmarks + #newlandmarks,), (3 + 2*(#landmarks + #newlandmarks,)*2
            eta with new landmarks appended, and its covariance
        """
        # TODO replace this with your own code
        n = P.shape[0]
        assert z.ndim == 1, "SLAM.add_landmarks: z must be a 1d array"

        numLmk = z.shape[0] // 2

        lmnew = np.empty_like(z)

        Gx = np.empty((numLmk * 2, 3))
        Rall = np.zeros((numLmk * 2, numLmk * 2))

        I2 = np.eye(2)  # Preallocate, used for Gx
        # For transforming landmark position into world frame
        sensor_offset_world = rotmat2d(eta[2]) @ self.sensor_offset
        sensor_offset_world_der = rotmat2d(
            eta[2] + np.pi / 2) @ self.sensor_offset  # Used in Gx

        for j in range(numLmk):
            ind = 2 * j
            inds = slice(ind, ind + 2)
            zj = z[inds]

            rj, thetaj = zj[0], zj[1]
            rot = rotmat2d(eta[2])  # rotmat in Gz

            # position of new landmark in world frame
            p_sensor = np.array([rj * np.cos(thetaj), rj * np.sin(thetaj)])
            lmnew[inds] = eta[0:2] + sensor_offset_world + rot @ p_sensor

            # Gx: d m_new / d [x, y, psi]
            Gx[inds, :2] = I2
            Gx[inds, 2] = (rot @ rotmat2d(np.pi / 2) @ p_sensor) + sensor_offset_world_der

            # Gz: d m_new / d [r, theta]  in sensor frame, then rotate to world with rot
            Gz_sensor = np.array([[np.cos(thetaj), -rj * np.sin(thetaj)],
                                  [np.sin(thetaj),  rj * np.cos(thetaj)]])
            Gz = rot @ Gz_sensor

            # transform measurement covariance from polar to cartesian coordinates
            Rall[inds, inds] = Gz @ self.R @ Gz.T

        assert len(lmnew) % 2 == 0, "SLAM.add_landmark: lmnew not even length"
        etaadded = np.hstack((eta, lmnew))  # append new landmarks to state vector

        # Block build P_new
        Padded = np.zeros((n + 2 * numLmk, n + 2 * numLmk))  # block diagonal of P_new
        Padded[:n, :n] = P

        # Cross-covariance between new landmarks and existing state
        Padded[n:, :n] = Gx @ P[:3, :n]     # top right corner of P_new
        Padded[:n, n:] = Padded[n:, :n].T   # transpose to enforce symmetry

        # Covariance of new landmarks block: Gx P_xx Gx^T + Rall
        Padded[n:, n:] = Gx @ P[:3, :3] @ Gx.T + Rall

        assert (
            etaadded.shape * 2 == Padded.shape
        ), "EKFSLAM.add_landmarks: calculated eta and P has wrong shape"
        assert np.allclose(
            Padded, Padded.T
        ), "EKFSLAM.add_landmarks: Padded not symmetric"
        assert np.all(
            np.linalg.eigvals(Padded) >= 0
        ), "EKFSLAM.add_landmarks: Padded not PSD"

        return etaadded, Padded

    def associate(
        self, z: np.ndarray, zpred: np.ndarray, H: np.ndarray, S: np.ndarray,
    ):  # -> Tuple[*((np.ndarray,) * 5)]:
        """Associate landmarks and measurements, and extract correct matrices for these.

        Parameters
        ----------
        z : np.ndarray,
            The measurements all in one vector
        zpred : np.ndarray
            Predicted measurements in one vector
        H : np.ndarray
            The measurement Jacobian matrix related to zpred
        S : np.ndarray
            The innovation covariance related to zpred

        Returns
        -------
        Tuple[*((np.ndarray,) * 5)]
            The extracted measurements, the corresponding zpred, H, S and the associations.

        Note
        ----
        See the associations are calculated  using JCBB. See this function for documentation
        of the returned association and the association procedure.
        """
        if self.do_asso:
            # Associate
            a = JCBB(z, zpred, S, self.alphas[0], self.alphas[1])

            # Extract associated measurements
            zinds = np.empty_like(z, dtype=bool)
            zinds[::2] = a > -1  # -1 means no association
            zinds[1::2] = zinds[::2]
            zass = z[zinds]

            # extract and rearange predicted measurements and cov
            zbarinds = np.empty_like(zass, dtype=int)
            zbarinds[::2] = 2 * a[a > -1]
            zbarinds[1::2] = 2 * a[a > -1] + 1

            zpredass = zpred[zbarinds]
            Sass = S[zbarinds][:, zbarinds]
            Hass = H[zbarinds]

            assert zpredass.shape == zass.shape
            assert Sass.shape == zpredass.shape * 2
            assert Hass.shape[0] == zpredass.shape[0]

            return zass, zpredass, Hass, Sass, a
        else:
            # should one do something her
            pass

    def update(
        self, eta: np.ndarray, P: np.ndarray, z: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        """Update eta and P with z, associating landmarks and adding new ones.

        Parameters
        ----------
        eta : np.ndarray
            the robot state and map concatenated
        P : np.ndarray
            the covariance of eta
        z : np.ndarray, shape=(#detections, 2)
            A set of measurements to create landmarks for

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, float, np.ndarray]
            updated eta, updated P, NIS, and the associations
        """
        # TODO replace this with your own code
        numLmk = (eta.size - 3) // 2
        assert (len(eta) - 3) % 2 == 0, "EKFSLAM.update: landmark lenght not even"

        if numLmk > 0:
            # Prediction and innovation covariance
            zpred = self.h(eta)
            H = self.h_jac(eta)

            # Big R for all current landmarks (block-diagonal, same R per landmark)
            Rbig = np.kron(np.eye(numLmk), self.R)
            S = H @ P @ H.T + Rbig
            assert (
                S.shape == zpred.shape * 2
            ), "EKFSLAM.update: wrong shape on either S or zpred"
            z = z.ravel()  # 2D -> flat

            # Perform data association
            za, zpred, Ha, Sa, a = self.associate(z, zpred, H, S)

            # No association could be made, so skip update
            if za.shape[0] == 0:
                etaupd = eta
                Pupd = P
                NIS = 1  # beware this one when analysing consistency.
            else:
                # Create the associated innovation
                v = za.ravel() - zpred  # za: 2D -> flat
                v[1::2] = utils.wrapToPi(v[1::2])

                # Kalman mean update (Sa is SPD; you can switch to Cholesky for speed)
                W = P @ Ha.T @ np.linalg.solve(Sa, np.eye(Sa.shape[0]))
                etaupd = eta + W @ v

                # Kalman cov update: Joseph form for stability
                jo = -W @ Ha
                jo[np.diag_indices(jo.shape[0])] += 1

                # Measurement noise for associated set (block-diag)
                Rass = np.kron(np.eye(za.shape[0] // 2), self.R)
                Pupd = jo @ P @ jo.T + W @ Rass @ W.T

                # NIS
                NIS = float(v.T @ np.linalg.solve(Sa, v))

                # Sanity checks
                assert np.allclose(Pupd, Pupd.T), "EKFSLAM.update: Pupd not symmetric"
                assert np.all(np.linalg.eigvals(Pupd) > 0), "EKFSLAM.update: Pupd not positive definite"

        else:  # All measurements are new landmarks,
            a = np.full(z.shape[0], -1)
            z = z.flatten()
            NIS = 1  # beware this one when analysing consistency.
            etaupd = eta
            Pupd = P

        # Create new landmarks if any is available
        if self.do_asso:
            is_new_lmk = a == -1
            if np.any(is_new_lmk):
                z_new_inds = np.empty_like(z, dtype=bool)
                z_new_inds[::2] = is_new_lmk
                z_new_inds[1::2] = is_new_lmk
                z_new = z[z_new_inds]
                etaupd, Pupd = self.add_landmarks(etaupd, Pupd, z_new)

        assert np.allclose(Pupd, Pupd.T), "EKFSLAM.update: Pupd must be symmetric"
        assert np.all(np.linalg.eigvals(Pupd) >= 0), "EKFSLAM.update: Pupd must be PSD"

        return etaupd, Pupd, NIS, a

    @classmethod
    def NEESes(cls, x: np.ndarray, P: np.ndarray, x_gt: np.ndarray,) -> np.ndarray:
        """Calculates the total NEES and the NEES for the substates
        Args:
            x (np.ndarray): The estimate
            P (np.ndarray): The state covariance
            x_gt (np.ndarray): The ground truth
        Raises:
            AssertionError: If any input is of the wrong shape, and if debug mode is on, certain numeric properties
        Returns:
            np.ndarray: NEES for [all, position, heading], shape (3,)
        """

        assert x.shape == (3,), f"EKFSLAM.NEES: x shape incorrect {x.shape}"
        assert P.shape == (3, 3), f"EKFSLAM.NEES: P shape incorrect {P.shape}"
        assert x_gt.shape == (
            3,), f"EKFSLAM.NEES: x_gt shape incorrect {x_gt.shape}"

        d_x = x - x_gt
        d_x[2] = utils.wrapToPi(d_x[2])
        assert (
            -np.pi <= d_x[2] <= np.pi
        ), "EKFSLAM.NEES: error heading must be between (-pi, pi)"

        d_p = d_x[0:2]
        P_p = P[0:2, 0:2]
        assert d_p.shape == (2,), "EKFSLAM.NEES: d_p must be 2 long"
        d_heading = d_x[2]  # Note: scalar
        assert np.ndim(
            d_heading) == 0, "EKFSLAM.NEES: d_heading must be scalar"
        P_heading = P[2, 2]  # Note: scalar
        assert np.ndim(
            P_heading) == 0, "EKFSLAM.NEES: P_heading must be scalar"

        # NB: Needs to handle both vectors and scalars! Additionally, must handle division by zero
        NEES_all = d_x @ (np.linalg.solve(P, d_x))
        NEES_pos = d_p @ (np.linalg.solve(P_p, d_p))
        try:
            NEES_heading = d_heading ** 2 / P_heading
        except ZeroDivisionError:
            NEES_heading = 1.0  # TODO: beware

        NEESes = np.array([NEES_all, NEES_pos, NEES_heading])
        NEESes[np.isnan(NEESes)] = 1.0  # We may divide by zero, # TODO: beware

        assert np.all(NEESes >= 0), "ESKF.NEES: one or more negative NEESes"
        return NEESes
