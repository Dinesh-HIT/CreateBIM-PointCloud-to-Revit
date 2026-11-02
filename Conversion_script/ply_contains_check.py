# no extra packages needed
ply = r"D:\MainCode\data\dmsr\damage\train_rebuild\test\meshes\ngp_100.ply"
with open(ply, "rb") as f:
    for _ in range(40):  # print first ~40 lines of the header
        line = f.readline().decode("utf-8", "ignore").rstrip()
        print(line)
        if line == "end_header":
            break
