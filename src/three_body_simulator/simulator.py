from typing import List
import numpy as np
import scipy as sp


def calculate_gravity(body1, body2):
    """二つの天体間の重力を計算する。

    Args:
        body1 (CelestialBody): 1つ目の天体, which must have 'position' and 'mass' attributes.
        body2 (CelestialBody): 2つ目の天体, which must have 'position' and 'mass' attributes.
    Returns:
        scipy.array: body1 が body2 から受ける重力ベクトル
    """
    G = 6.67430e-11
    r = sp.spatial.distance.euclidean(body1.position, body2.position)
    force_matunitude = G * body1.mass * body2.mass / r**2
    force_direction = -(body1.position - body2.position) / r
    return force_matunitude * force_direction


class CelestialBody:
    """天体を表すクラス"""

    def __init__(self, mass: float, position: np.array, velocity: np.array):
        """天体を初期化する。

        Args:
            mass (float): 天体の質量
            position (scipy.array): 天体の位置ベクトル
            velocity (scipy.array): 天体の速度ベクトル
        """
        self.mass = mass
        self.position = position
        self.velocity = velocity

    def update_velocity_and_position(self, bodies: List["CelestialBody"], dt: float):
        """天体の速度と位置を更新する。

        Args:
            bodies(List[CelestialBody]): 他の天体を含むリスト:
            dt (float): 時間の刻み幅
        """
        total_force = np.array([0.0, 0.0])
        for other_body in bodies:
            if other_body != self:
                total_force += calculate_gravity(self, other_body)

        def equation_motion(time: float, position: np.array, velocity: np.array):
            current_force = np.array([0.0, 0.0])
            pass
