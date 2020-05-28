import taichi as ti
import numpy as np

from ball_pytorch import RollingBallPytorchSim


ti.init(arch=ti.cpu, default_fp=ti.f32)

if __name__ == "__main__":

    gui = ti.GUI("ball", (256, 256))

    constants = {
        "radius": 0.05,
        "g": 9.8,
        "f": 0.007,
        "ro": 1000,
        "obstacles_elasticity": 0.8,
    }
    constants["volume"] = 4 * np.pi * (constants["radius"] ** 3) / 3
    constants["mass"] = constants["volume"] * constants["ro"]

    sim = RollingBallPytorchSim(
        constants=constants,
        sim_steps=20,
        max_time=1.,
        world_scale_coeff=10,
        grid_resolution=(32, 32),
        gui=gui,
        output_folder="./output",
    )
    sim.initial_speed[None] = [3.0, 1.0]

    for i in range(10):
        with ti.Tape(sim.loss):
            sim.run_simulation(
                initial_coordinate=[0.2, 0.5],
                attraction_coordinate=[0.5, 0.5],
                visualize=True,
            )
        #print(sim.velocity.grad[None][0])
        print(f'loss is {sim.loss[None]:.4f}', sim.initial_speed.grad[None][0], sim.initial_speed.grad[None][1])
        sim.initial_speed[None][0] -= 0.01 * sim.initial_speed.grad[None][0]
        sim.initial_speed[None][1] -= 0.01 * sim.initial_speed.grad[None][1]
