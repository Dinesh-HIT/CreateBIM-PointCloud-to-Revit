# -*- coding: utf-8 -*-
"""
Tooltip:
Displays a table of:
-> Pinned grid count
-> Unpinned grid count
->Monitored grid count
->Unmonitored grid count
->Total grid count
Version : V1 (2025.4)
Author : DS
"""



__title__ = "Create Roof"
__context__ = "doc-project"

# -*- coding: utf-8 -*-
import clr
import os
import json
import math

clr.AddReference('RevitAPI')
clr.AddReference('RevitServices')

from Autodesk.Revit.DB import *
from Autodesk.Revit.UI import TaskDialog
from RevitServices.Persistence import DocumentManager

# === Connect to Revit ===
doc = __revit__.ActiveUIDocument.Document

# === Locate JSON file (assumed in same folder as script) ===
script_dir = os.path.dirname(__file__)
json_path = os.path.join(script_dir, "revit_bounding_boxes.json")

if not os.path.exists(json_path):
    TaskDialog.Show("Missing File", "Could not find JSON file:\n{}".format(json_path))
    raise IOError("Bounding box JSON file not found.")

# === Load JSON data ===
with open(json_path, 'r') as f:
    bounding_boxes = json.load(f)

if not bounding_boxes or not isinstance(bounding_boxes, list):
    TaskDialog.Show("Error", "JSON format invalid or empty!")
    raise Exception("Invalid JSON structure.")

# === Find RoofType, WallType, and Level ===
roof_type = FilteredElementCollector(doc)\
    .OfCategory(BuiltInCategory.OST_Roofs)\
    .OfClass(RoofType)\
    .FirstElement()

wall_type = FilteredElementCollector(doc)\
    .OfCategory(BuiltInCategory.OST_Walls)\
    .OfClass(WallType)\
    .FirstElement()

level = FilteredElementCollector(doc)\
    .OfCategory(BuiltInCategory.OST_Levels)\
    .WhereElementIsNotElementType()\
    .FirstElement()

if not roof_type or not wall_type or not level:
    TaskDialog.Show("Error", "RoofType, WallType, or Level not found!")
    raise Exception("Missing project elements.")

# === Settings ===
apply_slope = True
slope_angle_degrees = 10
slope_radians = math.radians(slope_angle_degrees)
min_size_threshold = 2.0  # Minimum footprint size to accept bounding box

# === Start Transaction ===
t = Transaction(doc, "Auto Create Roofs and Walls from JSON")
t.Start()

created_roofs = 0
created_walls = 0
skipped = 0

try:
    for box in bounding_boxes:
        loc = box["location"]
        size = box["size"]

        # --- Filter tiny boxes ---
        if size["dx"] < min_size_threshold or size["dy"] < min_size_threshold:
            skipped += 1
            continue

        # --- Classify feature ---
        feature_type = "Unknown"
        if size["dz"] < 2.0 and size["dx"] > 2.0 and size["dy"] > 2.0:
            feature_type = "Roof"
        elif size["dz"] > 2.0 and (size["dx"] < 3.0 or size["dy"] < 3.0):
            feature_type = "Wall"
        elif size["dz"] < 4.0 and size["dx"] < 4.0 and size["dy"] < 4.0:
            feature_type = "Window/Door"

        print("Classified ID={} as {}".format(box.get("id", "Unknown"), feature_type))

        # --- Create footprint points ---
        p1 = XYZ(loc["x"] - size["dx"] / 2, loc["y"] - size["dy"] / 2, loc["z"])
        p2 = XYZ(loc["x"] + size["dx"] / 2, loc["y"] - size["dy"] / 2, loc["z"])
        p3 = XYZ(loc["x"] + size["dx"] / 2, loc["y"] + size["dy"] / 2, loc["z"])
        p4 = XYZ(loc["x"] - size["dx"] / 2, loc["y"] + size["dy"] / 2, loc["z"])

        # --- Depending on feature, create elements ---
        if feature_type == "Roof":
            curve_array = CurveArray()
            curve_array.Append(Line.CreateBound(p1, p2))
            curve_array.Append(Line.CreateBound(p2, p3))
            curve_array.Append(Line.CreateBound(p3, p4))
            curve_array.Append(Line.CreateBound(p4, p1))

            model_roof = doc.Create.NewFootPrintRoof(curve_array, level, roof_type)

            # Optional: Apply slope
            if apply_slope:
                footprint_roof = model_roof
                model_curve_set = footprint_roof.GetFootPrintCurves()
                for footprint_curve in model_curve_set:
                    footprint_roof.set_DefinesSlope(footprint_curve, True)
                    footprint_roof.set_SlopeAngle(footprint_curve, slope_radians)

            created_roofs += 1

        elif feature_type == "Wall":
            base_curve = Line.CreateBound(p1, p4)
            height = size["dz"]
            wall = Wall.Create(doc, base_curve, wall_type.Id, level.Id, height, 0.0, False, False)
            created_walls += 1

        else:
            skipped += 1

    t.Commit()
    TaskDialog.Show("Result", " Created {} Roofs\n Created {} Walls\n Skipped: {}".format(
        created_roofs, created_walls, skipped))

except Exception as e:
    t.RollBack()
    TaskDialog.Show("Error", "Error: {}".format(str(e)))
