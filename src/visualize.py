import matplotlib.pyplot as plt


def visualize_area(area_data, focused_node_id=None):
    """
    Visualizes the area data using matplotlib.

    Args:
        area_data (dict): The data returned by get_data_by_instruction.
        :param focused_node_id:
    """
    if not area_data:
        print("No data to visualize.")
        return

    if focused_node_id is None:
        focused_node_id = []

    nodes = area_data['area_nodes']
    links = area_data['area_links']
    pois = area_data['area_pois']
    route_osm_ids = area_data['instruction_data']['route']['osm_path']

    plt.figure(figsize=(10, 10))

    # Plot all nodes
    for node_id, node_data in nodes.items():
        if node_id in focused_node_id:
            # draw it in green
            plt.plot(node_data['lng'], node_data['lat'], 'co', markersize=10)
        else:
            plt.plot(node_data['lng'], node_data['lat'], 'ko', markersize=2)

    # Plot all links
    for link in links:
        node1 = nodes.get(link['source'])
        node2 = nodes.get(link['target'])
        if node1 and node2:
            plt.plot([node1['lng'], node2['lng']], [node1['lat'], node2['lat']], 'k-', linewidth=0.5)

    # Plot route nodes
    for idx, osm_id in enumerate(route_osm_ids):
        if osm_id in nodes:
            node = nodes[osm_id]
            if idx == 0:
                plt.plot(node['lng'], node['lat'], 'go', markersize=8)
            elif idx == len(route_osm_ids) - 1:
                plt.plot(node['lng'], node['lat'], 'ro', markersize=8)

            plt.plot(node['lng'], node['lat'], 'bo', markersize=4)

    # Plot route path
    for i in range(len(route_osm_ids) - 1):
        node1_id = route_osm_ids[i]
        node2_id = route_osm_ids[i + 1]
        if node1_id in nodes and node2_id in nodes:
            node1 = nodes[node1_id]
            node2 = nodes[node2_id]
            plt.plot([node1['lng'], node2['lng']], [node1['lat'], node2['lat']], 'b-', linewidth=2)

    # Plot POIs
    for poi_id, poi_data in pois.items():
        plt.plot(poi_data['lng'], poi_data['lat'], 'ro', markersize=5)
        if 'Jensen' in poi_data['tags']:
            plt.text(poi_data['lng'], poi_data['lat'], "F", fontsize=8)

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Area Visualization")
    plt.grid(True)
    plt.axis('equal')
    plt.show()
