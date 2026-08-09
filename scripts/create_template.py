"""Create task-extension files from the bundled LIBERO templates.

The templates are a starting point for custom tasks and still require task-
specific changes.
"""

import xml.etree.ElementTree as ET
from importlib.resources import as_file, files

from libero.libero.envs.textures import get_texture_file_list


def _template(name):
    return files("libero").joinpath("templates", name)


def create_problem_class_from_file(class_name):
    with as_file(_template("problem_class_template.py")) as template_source_file:
        with open(template_source_file) as f:
            lines = f.readlines()
    new_lines = []
    for line in lines:
        if "YOUR_CLASS_NAME" in line:
            line = line.replace("YOUR_CLASS_NAME", class_name)
        new_lines.append(line)
    with open(f"{class_name.lower()}.py", "w") as f:
        f.writelines(new_lines)
    print(f"Creating class {class_name} at the file: {class_name.lower()}.py")


def create_scene_xml_file(scene_name):
    """Create a scene XML template for further task-specific editing."""
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    with as_file(_template("scene_template.xml")) as template_source_file:
        tree = ET.parse(template_source_file, parser)
    root = tree.getroot()

    basic_elements = [
        ("Floor", "texplane"),
        ("Table", "tex-table"),
        ("Table legs", "tex-table-legs"),
        ("Walls", "tex-wall"),
    ]

    for element_name, texture_name in basic_elements:
        element = root.findall(f'.//texture[@name="{texture_name}"]')[0]
        type = None
        if "floor" in element_name.lower():
            type = "floor"
        elif "table" in element_name.lower():
            type = "table"
        elif "wall" in element_name.lower():
            type = "wall"
        # Pass a different texture_path to change where textures are discovered.
        texture_list = get_texture_file_list(type=type, texture_path="../")
        for i, (texture_name, texture_file_path) in enumerate(texture_list):
            print(f"[{i}]: {texture_name}")
        choice = int(input(f"Please select which texture to use for {element_name}: "))
        element.set("file", texture_list[choice][1])
    tree.write(f"{scene_name}.xml", encoding="utf-8")
    print(f"Creating scene {scene_name} at the file: {scene_name}.xml")
    print(
        "\n[Notice] Texture paths are relative and assume the scene XML will be "
        "placed under libero/libero/assets/scenes/."
    )
    return


def main():
    # use keyboard to select which file to create
    choices = [
        "problem_class",
        "scene",
        "object",
        "arena",
    ]

    for i, choice in enumerate(choices):
        print(f"[{i}]: {choice}")
    choice = int(input("Please select which file to create: "))

    if choices[choice] == "problem_class":
        # Ask user to specify the class name
        class_name = input("Please specify the class name: ")
        assert " " not in class_name, "space is not allowed in the naming"
        parts = class_name.split("_")
        class_name = "_".join([part.lower().capitalize() for part in parts])
        create_problem_class_from_file(class_name)
    elif choices[choice] == "scene":
        # Ask user to specify the scene name
        scene_name = input("Please specify the scene name: ")
        scene_name = scene_name.lower()
        assert " " not in scene_name, "space is not allowed in the naming"
        create_scene_xml_file(scene_name)


if __name__ == "__main__":
    main()
