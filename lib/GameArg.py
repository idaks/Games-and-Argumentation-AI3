import re
import pandas as pd
import subprocess
import os
from IPython.display import display, HTML
from collections import defaultdict

# import pygraphviz as pgv


# =================== General ===================
def reverse_edges(input_file_path: str, output_file_path: str):
    """
    This function reads a file containing graph edges,
    reverses the source and target nodes,
    and writes the modified edges to a new file.

    :param input_file_path: str, path to the input file.
    :param output_file_path: str, path to the output file.
    """
    try:
        with open(input_file_path, "r") as infile, open(
            output_file_path, "w"
        ) as outfile:
            for line in infile:
                # Extract source and target nodes by stripping the line
                line = line.strip()
                if line.startswith("edge(") and line.endswith(")."):
                    content = line[
                        len("edge(") : -2
                    ]  # remove 'edge(' and ').' from the line
                    source, target = content.split(",")
                    # Write reversed edge to the output file
                    outfile.write(f"edge({target},{source}).\n")

    except FileNotFoundError:
        print(f"{input_file_path} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


def read_edges_from_file(input_file, pred_name):
    edges = []
    pattern = re.compile(
        rf"{re.escape(pred_name)}\s*\(\s*\"?([^\",]+)\"?\s*,\s*\"?([^)\"]+)\"?\s*\)\."
    )
    try:
        with open(input_file, "r") as file:
            for line in file:
                match = pattern.match(line)
                if match:
                    source, target = match.groups()
                    edges.append((source, target))
    except FileNotFoundError:
        raise FileNotFoundError(f"{input_file} not found.")
    except Exception as e:
        raise RuntimeError(f"An error occurred: {str(e)}")
    return edges


def render_dot_to_png(dot_file_path, output_file_path):
    try:
        # Run the command to convert the DOT file to a PNG file
        subprocess.run(
            ["dot", "-Tpng", dot_file_path, "-o", output_file_path], check=True
        )

    except FileNotFoundError:
        print(f"File {dot_file_path} does not exist.")
    except subprocess.CalledProcessError:
        print(f"Error occurred while converting {dot_file_path} to PNG.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def run_command(cmd):
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )

    if result.returncode == 0:
        return result.stdout.decode()
    else:
        print(f"Error executing command: {cmd}")
        print(result.stderr.decode())
        return None


# ===================Visualization Specific===================
def create_plain_graph(input_file, pred_name, output_filename, arg=False):
    """
    Reads the edges from the input_file, creates a DataFrame, and writes
    the graph to a DOT file.
    :param input_file: str, the name of the input file
    :param pred_name: str, the predicate name to match in the input file
    :param output_filename: str, the name of the output DOT file
    without extension
    :param arg: bool, whether to add dir=back to the edges
    """

    edges = read_edges_from_file(input_file, pred_name)
    if not edges:  # Handle the case where no edges are found
        raise ValueError(
            f"No edges found in {input_file} for predicate {pred_name}."
        )

    edge_df = pd.DataFrame(edges, columns=["source", "target"])

    with open(f"{output_filename}", "w") as f:
        f.write("digraph G {\n")
        for _, row in edge_df.iterrows():
            dir_attr = (
                " [dir=back]" if arg else "[dir=forward]"
            )  # Add dir=back if arg is True
            f.write(f'    "{row["source"]}" -> "{row["target"]}"{dir_attr};\n')
        f.write("}\n")


# Group the properties of the edges
# the input would be lines of the graphviz file
def group_edges(input_list):
    edge_groups = defaultdict(list)
    result = []
    brace_count = 0  # Count of open braces to identify the last closing brace.

    # Parsing the input list.
    for index, line in enumerate(input_list):
        brace_count += line.count("{") - line.count("}")  # Update brace_count

        # If brace_count is zero after counting braces in the line, it means
        # we are at the last closing brace.
        if "}" in line and brace_count == 0:
            continue  # Don't add the closing } yet, we will add it at the end.

        # Extracting edges and their properties
        elif "->" in line:
            edge_prop_index = line.find("[")
            if edge_prop_index != -1:
                edge_str = line[:edge_prop_index].rstrip()
                prop_str = line[edge_prop_index:].rstrip().rstrip(";")
                edge_groups[prop_str].append(edge_str.rstrip(";").rstrip())

        # Preserving node, subgraph information, and any other lines.
        else:
            result.append(line)  # Preserve other lines as they are.

    # Grouping edges with the same properties
    for props, edges in edge_groups.items():
        result.append(f"  edge {props}\n")
        result.extend([f"    {edge};\n" for edge in edges])

    # Add the closing brace for the graph.
    result.append("}\n")

    return result


# Apply Color Schema to WFS
def apply_color_schema(
    dot_file_path,
    output_file_key,
    nodes_status,
    node_color,
    edge_color=None,
    subgraph=False,
):
    color_node_map = {
        "red": "#FFAAAA",
        "green": "#AAFFAA",
        "yellow": "#FFFFAA",
        "orange": "#bfefff",
        "blue": "#ffdaaf",
        "black": "#000000",
        "white": "#ffffff",
        "gray": "#b7b7b7",
    }

    color_edge_map = {
        "red": "#CC0000",
        "green": "#00BB00",
        "yellow": "#AAAA00",
        "gray": "#b7b7b7",
        "orange": "#cc8400",
        "blue": "#006ad1",
        "dark_gray": "#A9A9A9",
        "black": "#000000",
        "dark_yellow": "#000080",
    }

    with open(dot_file_path, "r") as file:
        lines = file.readlines()

    node_info = (
        "  node [shape=oval style=filled fontname=Helvetica fontsize=14]\n"
    )

    # Find the index to insert node_info, i.e., right after "digraph {"
    for idx, line in enumerate(lines):
        if line.strip().startswith("digraph"):
            insert_idx = idx + 1
            break
        else:
            raise ValueError("Improper dot file: 'digraph' not found")
    lines.insert(insert_idx, node_info)

    node_to_color = {}
    if nodes_status and node_color:
        if subgraph:
            g1_nodes = (
                []
            )  # You can decide the logic to split nodes between subgraphs
            g2_nodes = (
                []
            )  # You can decide the logic to split nodes between subgraphs
            for status, nodes in nodes_status.items():
                for node in nodes:
                    # Decide logic to append nodes to g1_nodes or g2_nodes.
                    third_key = list(node_color.keys())[2]
                    if status == third_key:
                        g2_nodes.append(node)
                    else:
                        g1_nodes.append(node)
            # print(g1_nodes, g2_nodes)
            # Creating subgraph cluster_g1 and cluster_g2 strings.
            subgraph_cluster_g1 = (
                "  subgraph cluster_g1{\n"
                '  label = "G1"; color = black; style ="dashed";\n'
            )
            subgraph_cluster_g2 = (
                "  subgraph cluster_g2{\n"
                '  label = "G2"; color = black; style ="dashed";\n'
            )

            for status, nodes in nodes_status.items():
                hex_color = color_node_map[node_color[status]]
                font_color = "#ffffff" if hex_color == "#000000" else "#000000"
                for node in nodes:
                    node_to_color[
                        node
                    ] = hex_color  # Map the node to its color.

                colored_nodes_line = (
                    '  node [fillcolor="'
                    + hex_color
                    + '" fontcolor="'
                    + font_color
                    + '"] '
                    + " ".join(nodes)
                    + ";\n"
                )

                if all(
                    node in g1_nodes for node in nodes
                ):  # Adjust based on your actual logic
                    subgraph_cluster_g1 += "  " + colored_nodes_line
                elif all(
                    node in g2_nodes for node in nodes
                ):  # Adjust based on your actual logic
                    subgraph_cluster_g2 += "  " + colored_nodes_line

            subgraph_cluster_g1 += "  }\n"  # Close subgraph cluster_g1
            subgraph_cluster_g2 += "  }\n"  # Close subgraph cluster_g2
            lines.insert(insert_idx + 1, subgraph_cluster_g1)
            lines.insert(insert_idx + 2, subgraph_cluster_g2)
            insert_idx += 2  # Adjusting the insert_idx after inserting

        else:
            for status, nodes in nodes_status.items():
                hex_color = color_node_map[node_color[status]]
                font_color = "#ffffff" if hex_color == "#000000" else "#000000"

                for node in nodes:
                    node_to_color[
                        node
                    ] = hex_color  # Map the node to its color.

                colored_nodes_line = (
                    '  node [fillcolor="'
                    + hex_color
                    + '" fontcolor="'
                    + font_color
                    + '"] '
                    + " ".join(nodes)
                    + ";\n"
                )
                insert_idx += 1
                lines.insert(insert_idx, colored_nodes_line)
    # stop execution if the edge_color is none
    if edge_color is None:
        output_file_path_dot = os.path.join(
            "graphs", f"{output_file_key}_node_colored.dot"
        )
        output_file_path_png = os.path.join(
            "graphs", f"{output_file_key}_node_colored.png"
        )

        # Use the output_file_path_dot as the file path to write to.
        try:
            with open(output_file_path_dot, "w") as file:
                file.writelines(lines)
        # Assuming lines is a list of strings to be written to the file.
        except Exception as e:
            print(f"An error occurred: {e}")

        # render the dot file to png
        render_dot_to_png(output_file_path_dot, output_file_path_png)

    else:
        # Manually add edge colors based on the node colors.
        for idx, line in enumerate(lines):
            if "->" in line:  # This line represents an edge.
                # Extract the source and target nodes of the edge.
                source_node, target_node = re.search(
                    r"([^ \[\]]+)\s*->\s*([^ \[\]]+)", line
                ).groups()

                # Get the color of source and target nodes.
                source_color = node_to_color.get(
                    source_node.replace('"', ""), "black"
                )  # Default to black if the node has no color.
                target_color = node_to_color.get(
                    target_node.replace('"', ""), "black"
                )  # Default to black if the node has no color.
                edge_color_default = "gray"
                source_color = next(
                    (
                        name
                        for name, color in color_node_map.items()
                        if color == source_color
                    ),
                    "black",
                )
                target_color = next(
                    (
                        name
                        for name, color in color_node_map.items()
                        if color == target_color
                    ),
                    "black",
                )
                if source_color and target_color:
                    # Determine the color of the edge based on the colors
                    selected_edge_color = edge_color.get(
                        (source_color, target_color), edge_color_default
                    )

                    # Map the selected edge color to its corresponding color
                    actual_edge_color = color_edge_map.get(
                        selected_edge_color, "#b7b7b7"
                    )
                    # Default to gray if there is no mapping.

                    # Check if attributes already exist and modify accordingly.
                    match = re.search(r"(\[.*\])", line)
                    if match:
                        # Append to the existing attribute section.
                        attributes = match.group(1)
                        attributes = attributes.rstrip("]")
                        new_attributes = (
                            f'{attributes}, color="{actual_edge_color}",'
                            f' style="solid"'
                        )
                        if selected_edge_color == "gray":
                            new_attributes = (
                                f'{attributes}, color="{actual_edge_color}",'
                                f' style="dashed"'
                            )
                        new_attributes += "]"
                        line_with_color = line.replace(
                            match.group(1), new_attributes
                        )
                    else:
                        # Create a new attribute section.
                        new_attributes = (
                            f'[color="{actual_edge_color}", style="solid"'
                        )
                        if selected_edge_color == "gray":
                            new_attributes = (
                                f'[color="{actual_edge_color}", style="dashed"'
                            )
                        new_attributes += "]"
                        line_with_color = (
                            line.rstrip("\n") + new_attributes + "\n"
                        )

                    lines[idx] = line_with_color

            # Optimize the edges
            lines_with_grouped_properties = group_edges(lines)

            # print(lines_with_grouped_properties)

            output_file_path_dot = os.path.join(
                "graphs", f"{output_file_key}_graph_colored.dot"
            )
            output_file_path_png = os.path.join(
                "graphs", f"{output_file_key}_graph_colored.png"
            )

            # Use the output_file_path_dot as the file path to write to.
            try:
                with open(output_file_path_dot, "w") as file:
                    file.writelines(lines_with_grouped_properties)
            # Assuming lines is a list of strings to be written to the file.
            except Exception as e:
                print(f"An error occurred: {e}")

            # render the dot file to png
            render_dot_to_png(output_file_path_dot, output_file_path_png)


# ===================WFS Related===================
# Get Node Status
def get_nodes_status(string, node_types):
    nodes_status = {}

    # Extract the true part and undefined part from the string
    true_part_match = re.search(r"True: {(.*?)}", string)
    true_part = true_part_match.group(1) if true_part_match else ""

    undefined_part_match = re.search(r"Undefined: {(.*?)}", string)
    undefined_part = (
        undefined_part_match.group(1) if undefined_part_match else ""
    )

    # Define a regex pattern to match nodes with
    # double quotes and special characters
    node_pattern = r'{}\(("[^"]+"|\w+)\)'

    # Go through each node type and find its
    # occurrences in the appropriate part
    for node_type in node_types:
        current_pattern = re.compile(node_pattern.format(node_type))
        nodes_list = current_pattern.findall(
            undefined_part
            if node_type in ["drawn", "undefined", "pk"]
            else true_part
        )
        # Clean the node names by removing the double quotes
        nodes_list = [node.strip('"') for node in nodes_list]
        nodes_status[node_type] = nodes_list

    return nodes_status


# Visualize Well_Founded Semantics
def visualize_wfs(
    plain_file,
    output_file_key,
    node_color,
    edge_color=None,
    arg=False,
    subgraph=False,
):
    temp_file_name = "wfs_compute.dlv"

    create_plain_graph(plain_file, "edge", "graphs/wfs_temp.dot", arg)

    try:
        facts_prep = (
            "e(X,Y):- edge(X,Y)." if not arg else "e(X,Y):- edge(Y,X)."
        )
        cal_wfs = """
        % Positions
        pos(X) :- e(X,_).
        pos(X) :- e(_,X).

        % Kernel
        status1(X) :- {}, status2(Y).
        status2(X) :- pos(X), not status1(X).
        status3(X) :- pos(X), not status1(X), not status2(X).
        """.format(
            "e(X,Y)" if not arg else "e(Y,X)"
        )

        keys = list(node_color.keys())
        for i in range(3):
            cal_wfs = cal_wfs.replace(f"status{i+1}", keys[i])

        with open(temp_file_name, "w+") as temp_file:
            temp_file.write(facts_prep + "\n" + cal_wfs)

        cmd_solve = f"dlv {plain_file} {temp_file_name} -wf"
        output = run_command(cmd_solve)

        if output:
            nodes_status = get_nodes_status(
                run_command(cmd_solve), node_types=list(node_color.keys())
            )
            apply_color_schema(
                "graphs/wfs_temp.dot",
                output_file_key,
                nodes_status,
                node_color,
                edge_color,
                subgraph,
            )
        else:
            print("No output received from command")

    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_name):
            try:
                os.remove(temp_file_name)
                os.remove("graphs/wfs_temp.dot")
            except Exception as e:
                print(f"Failed to delete {temp_file_name}. Error: {e}")


# ===================Stable Model Related===================
# stable Related
def extract_pws(input_string, predicates):
    # Extract parts enclosed by curly braces
    parts = re.findall(r"{(.*?)}", input_string)

    result = {}
    for idx, part in enumerate(parts, start=1):
        pw_name = f"pw{idx}"
        result[pw_name] = defaultdict(list)

        for predicate in predicates:
            nodes = re.findall(rf"{predicate}\((\w+)\)", part)
            if nodes:
                result[pw_name][predicate] = nodes

    return result


def visualize_stb(
    plain_file, output_file_key, node_color, edge_color=None, arg=False
):
    temp_file_name = "stable_compute.dlv"

    create_plain_graph(plain_file, "edge", "graphs/stb_temp.dot", arg)

    graph_name = "_graph_colored" if edge_color else "_node_colored"

    try:
        facts_prep = (
            "e(X,Y):- edge(X,Y)." if not arg else "e(X,Y):- edge(Y,X)."
        )
        cal_stb = """
        % Positions
        pos(X) :- e(X,_).
        pos(X) :- e(_,X).

        % Kernel
        status1(X) :- {}, status2(Y).
        status2(X) :- pos(X), not status1(X).
        """.format(
            "e(X,Y)" if not arg else "e(Y,X)"
        )

        keys = list(node_color.keys())
        for i in range(2):
            cal_stb = cal_stb.replace(f"status{i+1}", keys[i])

        with open(temp_file_name, "w+") as temp_file:
            temp_file.write(facts_prep + "\n" + cal_stb)

        cmd_solve = f"dlv {plain_file} {temp_file_name}"
        output = run_command(cmd_solve)
        image_files = []
        if output:
            pws = extract_pws(output, list(node_color.keys()))
            for pw, predicates_dict in pws.items():
                apply_color_schema(
                    "graphs/stb_temp.dot",
                    output_file_key + "_" + pw,
                    predicates_dict,
                    node_color,
                    edge_color,
                )
                image_files.append(
                    "graphs/"
                    + output_file_key
                    + "_"
                    + pw
                    + "{}.png".format(graph_name)
                )
        else:
            print("No output received from command")
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_name):
            try:
                os.remove(temp_file_name)
                os.remove("graphs/stb_temp.dot")
            except Exception as e:
                print(f"Failed to delete {temp_file_name}. Error: {e}")

    images_per_row = 4
    image_width = "300px"  # You can adjust the width to your preferred size
    html_str = ""

    for i, img_file in enumerate(image_files):
        # Extract pw_id from the file name
        pw_id = os.path.basename(img_file).split("_")[1]

        if i % images_per_row == 0:
            html_str += '<div style="text-align: center;">'  # Start a new row

        html_str += (
            f'<div style="display:inline-block; width: {image_width};">'
        )
        html_str += (
            f'<img src="{img_file}" style="width: 100%; height: auto;" />'
        )
        html_str += f"<div>{pw_id}</div></div>"  # Adding title

        if (
            i % images_per_row == images_per_row - 1
            or i == len(image_files) - 1
        ):
            html_str += "</div>"  # End the row

    display(HTML(html_str))
