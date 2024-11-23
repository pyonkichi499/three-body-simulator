import numpy as np

from src.three_body_simulator.simulator import CelestialBody, calculate_gravity


def test_calculate_gravity_no_force():
    """質量が0の場合は力が発生しないことを確認"""
    body1 = CelestialBody(mass=0, position=np.array([0, 0]), velocity=np.array([0, 0]))
    body2 = CelestialBody(mass=1, position=np.array([1, 0]), velocity=np.array([0, 0]))
    force = calculate_gravity(body1, body2)
    assert np.all(force == 0)


def test_calculate_gravity_equal_mass():
    """質量が等しい場合の重力を計算"""
    body1 = CelestialBody(mass=1, position=np.array([0, 0]), velocity=np.array([0, 0]))
    body2 = CelestialBody(mass=1, position=np.array([1, 0]), velocity=np.array([0, 0]))
    force = calculate_gravity(body1, body2)
    G = 6.67430e-11
    expected_force = np.array([-G, 0])
    np.testing.assert_allclose(force, expected_force)


def test_calculate_gravity_different_mass():
    """質量が異なる場合の重力を計算"""
    body1 = CelestialBody(mass=1, position=np.array([0, 0]), velocity=np.array([0, 0]))
    body2 = CelestialBody(mass=2, position=np.array([1, 0]), velocity=np.array([0, 0]))
    force = calculate_gravity(body1, body2)
    G = 6.67430e-11
    expected_force = np.array([-2 * G, 0])
    np.testing.assert_allclose(force, expected_force)


def test_update_velocity_and_position():
    """速度と位置の更新テスト"""
    body1 = CelestialBody(1.0, np.array([0.0, 0.0]), np.array([0.0, 0.0]))
    body2 = CelestialBody(1.0, np.array([1.0, 0.0]), np.array([0.0, 0.0]))

    bodies = [body2]
    dt = 0.1

    initial_position = body1.position.copy()
    body1.update_velocity_and_position(bodies, dt)

    # 位置が更新されていることを確認
    assert not np.array_equal(body1.position, initial_position)
