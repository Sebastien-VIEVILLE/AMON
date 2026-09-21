"""Particle Swarm Optimization solver for the AMON wind-farm benchmark."""

import argparse
import ast
import os
import time

import numpy as np


def _read_initial_solution(path):
	"""Read and validate the flattened ``[x0, y0, ...]`` layout format."""
	with open(path, "r", encoding="utf-8") as file:
		values = ast.literal_eval(file.readline())
	values = np.asarray(values, dtype=float)
	if values.ndim != 1 or values.size == 0 or values.size % 2:
		raise ValueError("The initial solution must contain an even number of coordinates")
	return values


def _fitness(eap, spacing, placing, diameter, scale):
	"""Turn the maximization objective and positive constraint violations into a score."""
	normalized_violation = spacing / max(2.0 * diameter, 1.0) + placing / max(scale, 1.0)
	return float(eap) - 1_000_000.0 * normalized_violation


def solve(
	param_file,
	x0_file=None,
	particles=12,
	iterations=30,
	seed=None,
	output_file=None,
):
	"""Optimize one instance and return the best layout and its evaluations.

	Parameters are deliberately small by default because each objective call runs
	a wake model. ``x0_file`` is used as the first particle when supplied.
	"""
	if particles < 2 or iterations < 1:
		raise ValueError("particles must be at least 2 and iterations must be positive")

	import constraints as cst
	import data as d
	import windfarm_eval
	import windfarm_setting as wf

	nb_wt, diameter, _, scale_factor, _, boundary_file, exclusion_file, _, _ = d.read_param_file(param_file)
	lower, upper, boundary, exclusions = wf.terrain_setting(
		boundary_file, exclusion_file, scale_factor=scale_factor
	)
	bounds_lower = np.tile(np.asarray(lower, dtype=float), nb_wt)
	bounds_upper = np.tile(np.asarray(upper, dtype=float), nb_wt)
	dimension = 2 * nb_wt
	span = float(np.linalg.norm(np.asarray(upper) - np.asarray(lower)))

	rng = np.random.default_rng(seed)
	positions = rng.uniform(bounds_lower, bounds_upper, size=(particles, dimension))
	if x0_file is not None:
		initial = _read_initial_solution(x0_file)
		if initial.size != dimension:
			raise ValueError(f"Expected {dimension} coordinates, got {initial.size}")
		positions[0] = np.clip(initial, bounds_lower, bounds_upper)

	velocities = rng.uniform(-0.1, 0.1, size=(particles, dimension)) * (bounds_upper - bounds_lower)
	personal_best = positions.copy()
	personal_scores = np.full(particles, -np.inf)
	global_best = None
	global_score = -np.inf
	global_evaluation = None
	evaluations = 0
	started = time.time()
	cache = {}
	os.makedirs("results", exist_ok=True)

	def evaluate(position):
		nonlocal evaluations
		candidate = tuple(np.round(position, 8))
		if candidate not in cache:
			eap, spacing, placing = windfarm_eval.windfarm_eval(param_file, list(candidate))
			score = _fitness(eap, spacing, placing, diameter, span)
			cache[candidate] = (score, float(eap), float(spacing), float(placing))
			evaluations += 1
		return cache[candidate]

	for iteration in range(iterations):
		inertia = 0.9 - 0.5 * iteration / max(iterations - 1, 1)
		for index in range(particles):
			evaluation = evaluate(positions[index])
			if evaluation[0] > personal_scores[index]:
				personal_scores[index] = evaluation[0]
				personal_best[index] = positions[index].copy()
			if evaluation[0] > global_score:
				global_score = evaluation[0]
				global_best = positions[index].copy()
				global_evaluation = evaluation

		random_personal = rng.random((particles, dimension))
		random_global = rng.random((particles, dimension))
		velocities = (
			inertia * velocities
			+ 1.7 * random_personal * (personal_best - positions)
			+ 1.7 * random_global * (global_best - positions)
		)
		velocities = np.clip(velocities, -(bounds_upper - bounds_lower), bounds_upper - bounds_lower)
		positions = np.clip(positions + velocities, bounds_lower, bounds_upper)

		if global_evaluation[2] <= 1e-9 and global_evaluation[3] <= 1e-9:
			break

	result = {
		"x_best": global_best.tolist(),
		"eap": global_evaluation[1],
		"spacing_constraint": global_evaluation[2],
		"placing_constraint": global_evaluation[3],
		"evaluations": evaluations,
		"iterations": iteration + 1,
		"time": time.time() - started,
	}
	if output_file is not None:
		output_directory = os.path.dirname(os.path.abspath(output_file))
		os.makedirs(output_directory, exist_ok=True)
		with open(output_file, "w", encoding="utf-8") as file:
			file.write(str(result["x_best"]))
	return result


def PSO_execution(param_file, x0_file=None, particles=12, iterations=30, seed=None):
	"""Compatibility entry point for running the solver from Python."""
	return solve(param_file, x0_file, particles, iterations, seed)


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("param_file", help="instance parameter file")
	parser.add_argument("x0_file", nargs="?", help="optional initial layout file")
	parser.add_argument("--particles", type=int, default=12)
	parser.add_argument("--iterations", type=int, default=30)
	parser.add_argument("--seed", type=int, default=None)
	parser.add_argument("--output", default=None, help="optional file for the best layout")
	args = parser.parse_args()

	result = solve(
		args.param_file,
		args.x0_file,
		args.particles,
		args.iterations,
		args.seed,
		args.output,
	)
	print(f"Best EAP = {result['eap']} GWh")
	print(f"Spacing constraint = {result['spacing_constraint']} m")
	print(f"Placing constraint = {result['placing_constraint']} m")
	print(f"Evaluations = {result['evaluations']}, time = {result['time']:.2f} s")


if __name__ == "__main__":
	main()
