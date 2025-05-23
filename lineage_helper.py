import sqlite3
import networkx as nx
import pydot

def extract_lineage_graph(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]

    # Build a directed graph of foreign key relationships
    G = nx.DiGraph()

    for table in tables:
        cursor.execute(f"PRAGMA foreign_key_list({table});")
        fks = cursor.fetchall()
        for fk in fks:
            from_col = fk[3]
            to_table = fk[2]
            to_col = fk[4]
            G.add_edge(table, to_table, label = f"{from_col} -> {to_col}")
            # from_node = f"{table}.{from_col}"
            # to_node = f"{to_table}.{to_col}"
            # G.add_edge(from_node, to_node, label=f"{from_col} -> {to_col}")


    conn.close()
    return G

def get_table_lineage_path(db_path, start_table):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]

    G = nx.DiGraph()

    for table in tables:
        cursor.execute(f"PRAGMA foreign_key_list({table});")
        fks = cursor.fetchall()
        for fk in fks:
            from_col = fk[3]
            to_table = fk[2]
            to_col = fk[4]
            G.add_edge(table, to_table, label=f"{from_col} -> {to_col}")

    conn.close()

    # Trace downstream lineage from start_table
    subgraph_nodes = nx.descendants(G, start_table)
    subgraph_nodes.add(start_table)
    subG = G.subgraph(subgraph_nodes).copy()

    return subG

def get_graphviz_layout_with_ranksep(G, prog="dot", ranksep=2.0):
    # Convert NetworkX graph to Pydot
    P = nx.nx_pydot.to_pydot(G)

    # Set ranksep for vertical spacing
    P.set("ranksep", str(ranksep))

    # Generate new DOT string and convert back to a NetworkX graph
    dot_str = P.to_string()
    # Now get layout directly from this DOT string
    pos = nx.nx_pydot.graphviz_layout(nx.nx_pydot.from_pydot(pydot.graph_from_dot_data(dot_str)[0]), prog=prog)

    return pos


def extract_column_level_lineage(db_path):
    """
    Extracts column-level technical lineage for business lineage derivation.
    Returns a NetworkX DiGraph with nodes like 'Invoice.CustomerId'.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]

    G = nx.DiGraph()
    for table in tables:
        cursor.execute(f"PRAGMA foreign_key_list({table});")
        for fk in cursor.fetchall():
            from_node = f"{table}.{fk[3]}"
            to_node = f"{fk[2]}.{fk[4]}"
            G.add_edge(from_node, to_node)
    
    conn.close()
    return G



def collapse_to_business_lineage(G):
    business_edges = set()
    for src, dst in G.edges():
        src_table = src.split('.')[0]
        dst_table = dst.split('.')[0]
        if src_table != dst_table:
            business_edges.add((src_table, dst_table))
    return nx.DiGraph(list(business_edges))

