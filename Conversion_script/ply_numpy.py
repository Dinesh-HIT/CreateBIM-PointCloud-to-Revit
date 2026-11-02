import numpy as np

def ply_ascii_to_numpy(ply_file):
    points = []
    header_finished = False

    with open(ply_file, "r") as file:
        for line in file:
            line = line.strip()

            # Skip header
            if line == "end_header":
                header_finished = True
                continue

            if not header_finished:
                continue

            # Read x, y, z values
            values = line.split()
            x, y, z = map(float, values[:3])
            points.append([x, y, z])

    return np.array(points)


# Example use
points = ply_ascii_to_numpy(r"F:\CreateBIM\dataset\final_dataset\Experiment\experiment_voting_transfer.ply")

print(points.shape)
print(points[:5])

np.save("points.npy", points)
np.save(r"F:\CreateBIM\dataset\final_dataset\Experiment\experiment_voting_transfer_NPY.npy",  points)