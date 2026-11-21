# Design Diagnosis Device

Mechanical design of the prototype point-of-care reader (enclosure, µPAD slider, camera neck with LED holder,
light-sensor holder). It is the hardware on which `Diagnosis Application/` runs. The construction and function of the
device is shown in Supplementary Figure S6 of the article; the component list is `BOM.csv` (Supplementary Table S17).

## Content

| Path | Content | Format |
|---|---|---|
| `cad/major_revisi.FCStd` | native design project (FreeCAD 1.0.1) with five bodies | `.FCStd` |
| `stl/major_revisi_assembled.stl` | all parts in assembled position, for orientation only (356 x 189 x 180 mm) | `.stl` |
| `stl/major_revisi_body.stl` | main body: display housing and camera neck with rails and holders | `.stl` |
| `stl/major_revisi_cam_close.stl` | vertical cover of the camera neck; keeps ambient light away from the sample so that only the LEDs illuminate it | `.stl` |
| `stl/major_revisi_slider.stl` | sliding board that carries the µPAD along the rails on both sides of the camera neck | `.stl` |
| `stl/major_revisi_tab_btm_close.stl` | bottom cover behind the camera; the opening between the two sections eases cabling to the Raspberry Pi 5 | `.stl` |
| `stl/major_revisi_ex_btm_close.stl` | bottom cover flush with the touchscreen | `.stl` |
| `print_3mf/Body_Neptune4Max_Cura_layer0.20.3mf` | print-ready project of the main body (UltiMaker Cura 5.10.0) | `.3mf` |
| `print_3mf/SmallParts_Neptune4Max_Cura_layer0.15.3mf` | print-ready project of the four smaller parts (UltiMaker Cura 5.10.0) | `.3mf` |
| `images/` | FreeCAD screenshot of the model and the device construction figure (Supplementary Figure S6) | `.png`, `.jpg` |
| `BOM.csv` | bill of materials (component, specification, source) | CSV |

## Fabrication

* Design: FreeCAD 1.0.1. Each body of the project was exported to STL.
* Printer and material: ELEGOO Neptune 4 Pro, white PLA+ filament, 1.75 mm, 0.4 mm nozzle.
* Main body (`Body_...3mf`): layer height 0.20 mm, infill 15 % (grid), top and bottom thickness 0.8 mm, brim, supports.
* Small parts (`SmallParts_...3mf`): layer height 0.15 mm, top and bottom thickness 0.9 mm, supports.
* The 3MF files carry the Cura settings that differ from the printer profile; temperatures (bed 60 C, nozzle 200 C) come
  from the generic PLA profile.
* The main body measures 356 mm along its longest side.
* The two 3MF project files were exported from Cura with the machine definition `elegoo_neptune_4max`; use them for the print settings (layer height, infill, supports) and select the Neptune 4 Pro profile for your printer before slicing.
* To print another part, load its STL in the slicer with the same profile, or open the 3MF files above and replace the model.

Files are released under CC BY 4.0 (see `../LICENSE-DATA.md`).
