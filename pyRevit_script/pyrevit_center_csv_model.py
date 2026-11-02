# -*- coding: utf-8 -*-
import csv
import os

from Autodesk.Revit.DB import (
    FilteredElementCollector,
    BuiltInCategory,
    LocationPoint,
    FamilyInstance
)

doc = __revit__.ActiveUIDocument.Document

OUTPUT_CSV = r"F:\CreateBIM\dataset\validation\reference_opening_centers.csv"

#OUTPUT_CSV = r"F:\CreateBIM\dataset\validation\automatic_opening_centers.csv"

def get_symbol_name(elem):
    try:
        family_name = elem.Symbol.Family.Name if elem.Symbol and elem.Symbol.Family else ""
    except:
        family_name = ""
    try:
        type_name = elem.Name
    except:
        type_name = ""
    return family_name, type_name

def get_instance_mark(elem):
    try:
        p = elem.LookupParameter("Mark")
        if p and p.HasValue:
            return p.AsString()
    except:
        pass
    return ""

def get_level_name(elem):
    try:
        level = doc.GetElement(elem.LevelId)
        if level:
            return level.Name
    except:
        pass
    return ""

def collect_family_instances(bic, category_name):
    results = []
    collector = FilteredElementCollector(doc).OfCategory(bic).WhereElementIsNotElementType()

    for elem in collector:
        if not isinstance(elem, FamilyInstance):
            continue

        loc = elem.Location
        if not isinstance(loc, LocationPoint):
            continue

        pt = loc.Point
        family_name, type_name = get_symbol_name(elem)
        mark = get_instance_mark(elem)
        level_name = get_level_name(elem)

        results.append({
            "unique_id": str(elem.UniqueId),
            "element_id": str(elem.Id.IntegerValue),
            "category": str(category_name),
            "family": str(family_name),
            "type": str(type_name),
            "mark": str(mark),
            "level": str(level_name),
            "x_ft": pt.X,
            "y_ft": pt.Y,
            "z_ft": pt.Z,
            "x_m": pt.X * 0.3048,
            "y_m": pt.Y * 0.3048,
            "z_m": pt.Z * 0.3048,
        })

    return results

def main():
    out_dir = os.path.dirname(OUTPUT_CSV)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    rows = []
    rows.extend(collect_family_instances(BuiltInCategory.OST_Windows, "Window"))
    rows.extend(collect_family_instances(BuiltInCategory.OST_Doors, "Door"))

    if not rows:
        print("No door/window instances found.")
        return

    fieldnames = [
        "unique_id", "element_id", "category", "family", "type", "mark", "level",
        "x_ft", "y_ft", "z_ft", "x_m", "y_m", "z_m"
    ]

    with open(OUTPUT_CSV, "wb") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print("Export completed: {}".format(OUTPUT_CSV))
    print("Exported {} opening instances.".format(len(rows)))

if __name__ == "__main__":
    main()