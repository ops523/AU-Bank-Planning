import io
import math
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pydeck as pdk
import requests
import streamlit as st
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from sklearn.cluster import KMeans

# ... rest of your code continues here

REQUIRED_COLUMNS = {"branch_name", "latitude", "longitude"}


def haversine_km(lat1, lon1, lat2, lon2):
    radius_km = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius_km * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def build_haversine_matrix(df):
    coords = df[["latitude", "longitude"]].to_numpy(float)
    n = len(coords)
    matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i, j] = haversine_km(coords[i][0], coords[i][1], coords[j][0], coords[j][1])
    return matrix


def build_osrm_matrix(df, osrm_base_url):
    coords = [f"{row.longitude},{row.latitude}" for row in df.itertuples()]
    coord_text = ";".join(coords)
    url = f"{osrm_base_url.rstrip('/')}/table/v1/driving/{coord_text}"
    response = requests.get(url, params={"annotations": "distance"}, timeout=60)
    response.raise_for_status()
    data = response.json()
    return np.array(data["distances"], dtype=float) / 1000.0


def build_ors_matrix(df, ors_api_key):
    locations = [[row.longitude, row.latitude] for row in df.itertuples()]
    response = requests.post(
        "https://api.openrouteservice.org/v2/matrix/driving-car",
        headers={"Authorization": ors_api_key, "Content-Type": "application/json"},
        json={"locations": locations, "metrics": ["distance"], "units": "km"},
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    return np.array(data["distances"], dtype=float)


def get_distance_matrix(df, provider, osrm_base_url, ors_api_key):
    if len(df) <= 1:
        return np.zeros((len(df), len(df)), dtype=float), "Single branch, no travel distance"
    if provider == "OSRM":
        return build_osrm_matrix(df, osrm_base_url), "OSRM road distance"
    if provider == "openrouteservice":
        return build_ors_matrix(df, ors_api_key), "openrouteservice road distance"
    return build_haversine_matrix(df), "Haversine straight-line distance"


def cluster_branches(df, number_of_clusters):
    working = df.copy()
    coords = working[["latitude", "longitude"]].to_numpy(float)
    model = KMeans(n_clusters=number_of_clusters, n_init=20, random_state=42)
    working["cluster_id"] = model.fit_predict(coords) + 1
    centers = pd.DataFrame(model.cluster_centers_, columns=["center_latitude", "center_longitude"])
    centers["cluster_id"] = centers.index + 1
    return working, centers


def allocate_teams(clustered_df, number_of_teams, estimated_days_per_branch=1):
    """Initial cluster assignment + dynamic rebalancing for even workload"""
    # Initial greedy assignment by clusters
    cluster_workload = (
        clustered_df.groupby("cluster_id")
        .size()
        .reset_index(name="branch_count")
    )
    cluster_workload["estimated_execution_days"] = cluster_workload["branch_count"] * estimated_days_per_branch
    cluster_workload = cluster_workload.sort_values("estimated_execution_days", ascending=False)

    team_loads = {team: 0 for team in range(1, number_of_teams + 1)}
    assignments = []

    for row in cluster_workload.itertuples():
        team_id = min(team_loads, key=team_loads.get)
        assignments.append({
            "cluster_id": row.cluster_id,
            "team_id": team_id,
            "cluster_branch_count": row.branch_count,
            "cluster_estimated_execution_days": row.estimated_execution_days,
        })
        team_loads[team_id] += row.estimated_execution_days

    df = clustered_df.merge(pd.DataFrame(assignments), on="cluster_id", how="left")

    # === NEW: Dynamic Rebalancing ===
    df = balance_team_workload(df, estimated_days_per_branch, number_of_teams)

    return df

    def balance_team_workload(df, estimated_days_per_branch=1, number_of_teams=4):
    """Rebalance branches from heavy teams to light teams using proximity"""
    df = df.copy()
    
    while True:
        # Calculate current load per team
        team_load = df.groupby("team_id").size() * estimated_days_per_branch
        max_load = team_load.max()
        min_load = team_load.min()
        
        # Stop if workload is balanced (difference ≤ 1 branch)
        if max_load - min_load <= estimated_days_per_branch:
            break
            
        heavy_teams = team_load[team_load == max_load].index.tolist()
        light_teams = team_load[team_load == min_load].index.tolist()
        
        if not heavy_teams or not light_teams:
            break
            
        # Try to move one branch from heavy to light team
        moved = False
        for heavy_team in heavy_teams:
            heavy_branches = df[df["team_id"] == heavy_team]
            
            for light_team in light_teams:
                light_branches = df[df["team_id"] == light_team]
                
                # Find closest branch from heavy team to light team's branches
                best_branch = None
                best_dist = float('inf')
                
                for _, h_row in heavy_branches.iterrows():
                    for _, l_row in light_branches.iterrows():
                        dist = haversine_km(
                            h_row["latitude"], h_row["longitude"],
                            l_row["latitude"], l_row["longitude"]
                        )
                        if dist < best_dist:
                            best_dist = dist
                            best_branch = h_row.name
                
                if best_branch is not None and best_dist < 150:  # Max 150km move threshold
                    df.loc[best_branch, "team_id"] = light_team
                    moved = True
                    break
            if moved:
                break
        if not moved:
            break  # Cannot improve further
    
    return df

    
    return clustered_df.merge(pd.DataFrame(assignments), on="cluster_id", how="left")


def solve_route(distance_matrix_km, start_index=0):
    n = len(distance_matrix_km)
    if n <= 1:
        return list(range(n)), 0.0

    scaled = (distance_matrix_km * 1000).astype(int)
    manager = pywrapcp.RoutingIndexManager(n, 1, start_index)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(scaled[from_node][to_node])

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.seconds = 10

    solution = routing.SolveWithParameters(search_parameters)
    if not solution:
        return list(range(n)), float(distance_matrix_km.sum())

    route = []
    route_distance_m = 0
    index = routing.Start(0)
    while not routing.IsEnd(index):
        node = manager.IndexToNode(index)
        route.append(node)
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        route_distance_m += routing.GetArcCostForVehicle(previous_index, index, 0)

    return route, route_distance_m / 1000.0


def create_routes(allocated_df, provider, osrm_base_url, ors_api_key):
    route_rows = []
    route_summary_rows = []
    distance_matrix_rows = []

    for (team_id, cluster_id), group in allocated_df.groupby(["team_id", "cluster_id"]):
        group = group.reset_index(drop=True)
        matrix, distance_source = get_distance_matrix(group, provider, osrm_base_url, ors_api_key)
        route_indexes, route_km = solve_route(matrix)

        for from_index, from_branch in group.iterrows():
            for to_index, to_branch in group.iterrows():
                distance_matrix_rows.append(
                    {
                        "team_id": team_id,
                        "cluster_id": cluster_id,
                        "from_branch": from_branch["branch_name"],
                        "to_branch": to_branch["branch_name"],
                        "from_latitude": from_branch["latitude"],
                        "from_longitude": from_branch["longitude"],
                        "to_latitude": to_branch["latitude"],
                        "to_longitude": to_branch["longitude"],
                        "distance_km": round(float(matrix[from_index][to_index]), 2),
                        "distance_source": distance_source,
                    }
                )

        for sequence, local_index in enumerate(route_indexes, start=1):
            branch = group.iloc[local_index].to_dict()
            previous_local_index = route_indexes[sequence - 2] if sequence > 1 else local_index
            leg_km = matrix[previous_local_index][local_index] if sequence > 1 else 0.0
            route_rows.append(
                {
                    "team_id": team_id,
                    "cluster_id": cluster_id,
                    "route_sequence": sequence,
                    "leg_from_previous_km": round(float(leg_km), 2),
                    "branch_name": branch["branch_name"],
                    "latitude": branch["latitude"],
                    "longitude": branch["longitude"],
                    "city": branch.get("city", ""),
                    "state": branch.get("state", ""),
                    "pincode": branch.get("pincode", ""),
                }
            )

        route_summary_rows.append(
            {
                "team_id": team_id,
                "cluster_id": cluster_id,
                "branches": len(group),
                "estimated_route_km": round(route_km, 2),
                "distance_source": distance_source,
            }
        )

    return pd.DataFrame(route_rows), pd.DataFrame(route_summary_rows), pd.DataFrame(distance_matrix_rows)


def create_daily_calendar(
    route_df,
    start_date,
    walls_per_branch,
    max_walls_per_team_per_day,
    productivity_factor,
    wall_availability_factor,
    max_daily_travel_km,
):
    calendar_rows = []
    effective_walls_per_day = max(0.25, max_walls_per_team_per_day * productivity_factor * wall_availability_factor)
    for team_id, team_routes in route_df.groupby("team_id"):
        team_routes = team_routes.sort_values(["cluster_id", "route_sequence"]).reset_index(drop=True)
        team_day_number = 0
        for _, row in team_routes.iterrows():
            leg_km = float(row.get("leg_from_previous_km", 0) or 0)
            travel_buffer_days = max(0, math.ceil(leg_km / max_daily_travel_km) - 1) if max_daily_travel_km > 0 else 0

            for buffer_index in range(travel_buffer_days):
                calendar_rows.append(
                    {
                        "date": start_date + timedelta(days=team_day_number),
                        "team_id": team_id,
                        "day_number": team_day_number + 1,
                        "activity": "Travel buffer",
                        "visit_order": "",
                        "cluster_id": row["cluster_id"],
                        "branch_name": row["branch_name"],
                        "planned_walls": 0.0,
                        "walls_per_branch": walls_per_branch,
                        "effective_walls_per_day": round(effective_walls_per_day, 2),
                        "leg_from_previous_km": round(leg_km, 2) if buffer_index == 0 else 0.0,
                        "city": row.get("city", ""),
                        "state": row.get("state", ""),
                        "pincode": row.get("pincode", ""),
                        "latitude": row["latitude"],
                        "longitude": row["longitude"],
                    }
                )
                team_day_number += 1

            remaining_walls = float(walls_per_branch)
            execution_day = 1
            while remaining_walls > 0:
                planned_walls = min(effective_walls_per_day, remaining_walls)
                calendar_rows.append(
                    {
                        "date": start_date + timedelta(days=team_day_number),
                        "team_id": team_id,
                        "day_number": team_day_number + 1,
                        "activity": "Branch execution",
                        "visit_order": execution_day,
                        "cluster_id": row["cluster_id"],
                        "branch_name": row["branch_name"],
                        "planned_walls": round(planned_walls, 2),
                        "walls_per_branch": walls_per_branch,
                        "effective_walls_per_day": round(effective_walls_per_day, 2),
                        "leg_from_previous_km": round(leg_km, 2) if execution_day == 1 and travel_buffer_days == 0 else 0.0,
                        "city": row.get("city", ""),
                        "state": row.get("state", ""),
                        "pincode": row.get("pincode", ""),
                        "latitude": row["latitude"],
                        "longitude": row["longitude"],
                    }
                )
                remaining_walls = round(remaining_walls - planned_walls, 6)
                execution_day += 1
                team_day_number += 1
    return pd.DataFrame(calendar_rows)


def create_costing(route_summary_df, cost_per_km, daily_team_cost, calendar_df):
    team_days = calendar_df.groupby("team_id")["date"].nunique().reset_index(name="working_days")
    route_km = route_summary_df.groupby("team_id")["estimated_route_km"].sum().reset_index(name="total_route_km")
    planned_walls = calendar_df.groupby("team_id")["planned_walls"].sum().reset_index(name="planned_walls")
    branch_days = (
        calendar_df[calendar_df["activity"] == "Branch execution"]
        .groupby("team_id")
        .size()
        .reset_index(name="branch_execution_days")
    )
    travel_buffer_days = (
        calendar_df[calendar_df["activity"] == "Travel buffer"]
        .groupby("team_id")
        .size()
        .reset_index(name="travel_buffer_days")
    )
    costing = team_days.merge(route_km, on="team_id", how="left")
    costing = costing.merge(planned_walls, on="team_id", how="left")
    costing = costing.merge(branch_days, on="team_id", how="left")
    costing = costing.merge(travel_buffer_days, on="team_id", how="left")
    costing[["branch_execution_days", "travel_buffer_days"]] = costing[
        ["branch_execution_days", "travel_buffer_days"]
    ].fillna(0)
    costing["travel_cost"] = costing["total_route_km"] * cost_per_km
    costing["team_cost"] = costing["working_days"] * daily_team_cost
    costing["total_cost"] = costing["travel_cost"] + costing["team_cost"]
    return costing.round(2)


def create_dashboard_df(
    coordinates_df,
    clustered_df,
    route_summary_df,
    costing_df,
    max_project_days,
    project_days_required,
    deadline_status,
    minimum_teams_for_execution,
    minimum_teams_after_travel,
):
    return pd.DataFrame(
        [
            {"metric": "Total branches", "value": len(coordinates_df)},
            {"metric": "Total clusters", "value": clustered_df["cluster_id"].nunique()},
            {"metric": "Total teams", "value": costing_df["team_id"].nunique()},
            {"metric": "Total route km", "value": round(route_summary_df["estimated_route_km"].sum(), 2)},
            {"metric": "Total planned walls", "value": round(costing_df["planned_walls"].sum(), 2)},
            {"metric": "Total team working days", "value": int(costing_df["working_days"].sum())},
            {"metric": "Project days required", "value": project_days_required},
            {"metric": "Max project days", "value": max_project_days},
            {"metric": "Deadline status", "value": deadline_status},
            {"metric": "Minimum teams for execution deadline", "value": minimum_teams_for_execution},
            {"metric": "Minimum teams after travel buffers", "value": minimum_teams_after_travel},
            {"metric": "Total travel cost", "value": round(costing_df["travel_cost"].sum(), 2)},
            {"metric": "Total team cost", "value": round(costing_df["team_cost"].sum(), 2)},
            {"metric": "Total cost", "value": round(costing_df["total_cost"].sum(), 2)},
        ]
    )


def color_for_id(value):
    palette = [
        [31, 119, 180, 180],
        [255, 127, 14, 180],
        [44, 160, 44, 180],
        [214, 39, 40, 180],
        [148, 103, 189, 180],
        [140, 86, 75, 180],
        [227, 119, 194, 180],
        [127, 127, 127, 180],
        [188, 189, 34, 180],
        [23, 190, 207, 180],
    ]
    return palette[int(value - 1) % len(palette)]


def render_cluster_map(clustered_df):
    map_df = clustered_df.copy()
    map_df["cluster_color"] = map_df["cluster_id"].apply(color_for_id)
    view_state = pdk.ViewState(
        latitude=float(map_df["latitude"].mean()),
        longitude=float(map_df["longitude"].mean()),
        zoom=5,
        pitch=0,
    )
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position="[longitude, latitude]",
        get_fill_color="cluster_color",
        get_radius=7000,
        pickable=True,
    )
    st.pydeck_chart(
        pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip={"text": "{branch_name}\nCluster {cluster_id}\nTeam {team_id}"},
        )
    )


def render_route_map(route_df):
    route_map_df = route_df.copy()
    route_map_df["team_color"] = route_map_df["team_id"].apply(color_for_id)
    route_map_df["path"] = None
    path_rows = []
    for (team_id, cluster_id), group in route_map_df.groupby(["team_id", "cluster_id"]):
        group = group.sort_values("route_sequence")
        coords = group[["longitude", "latitude"]].values.tolist()
        if len(coords) > 1:
            path_rows.append(
                {
                    "team_id": team_id,
                    "cluster_id": cluster_id,
                    "path": coords,
                    "team_color": color_for_id(team_id),
                }
            )
    view_state = pdk.ViewState(
        latitude=float(route_map_df["latitude"].mean()),
        longitude=float(route_map_df["longitude"].mean()),
        zoom=5,
        pitch=0,
    )
    layers = [
        pdk.Layer(
            "ScatterplotLayer",
            data=route_map_df,
            get_position="[longitude, latitude]",
            get_fill_color="team_color",
            get_radius=7000,
            pickable=True,
        )
    ]
    if path_rows:
        layers.append(
            pdk.Layer(
                "PathLayer",
                data=pd.DataFrame(path_rows),
                get_path="path",
                get_color="team_color",
                width_min_pixels=3,
            )
        )
    st.pydeck_chart(
        pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            tooltip={"text": "{branch_name}\nTeam {team_id}\nSequence {route_sequence}"},
        )
    )


def export_workbook(sheets):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
            worksheet = writer.sheets[sheet_name[:31]]
            for column_index, column_name in enumerate(df.columns):
                width = max(12, min(35, len(str(column_name)) + 4))
                worksheet.set_column(column_index, column_index, width)
    output.seek(0)
    return output


def clean_input(df):
    df = df.copy()
    df.columns = [str(col).strip().lower().replace(" ", "_") for col in df.columns]
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])
    df = df[(df["latitude"].between(6, 38)) & (df["longitude"].between(68, 98))]
    return df.reset_index(drop=True)


st.set_page_config(page_title="AU Bank Branch Deployment Planner", layout="wide")
st.title("AU Bank Branch Deployment Planner")

uploaded_file = st.file_uploader("Upload the branch coordinate Excel file", type=["xlsx", "xls", "csv"])

with st.sidebar:
    st.header("Planning Inputs")
    max_project_days = st.number_input("Max project days", min_value=1, max_value=365, value=30, step=1)
    number_of_clusters = st.number_input("Clusters", min_value=2, max_value=100, value=8, step=1)
    number_of_teams = st.number_input("Teams", min_value=1, max_value=20, value=4, step=1)
    deployment_start = st.date_input("Deployment start date", value=date.today())
    walls_per_branch = st.number_input("Walls per branch", min_value=1.0, max_value=10.0, value=2.0, step=0.5)
    max_walls_per_team_per_day = st.number_input("Max walls per team per day", min_value=0.5, max_value=5.0, value=2.0, step=0.5)
    weather_productivity_percent = st.slider("Weather/productivity factor", min_value=25, max_value=100, value=100, step=5)
    wall_availability_percent = st.slider("Wall availability factor", min_value=25, max_value=100, value=100, step=5)
    max_daily_travel_km = st.number_input("Max practical travel km per day", min_value=25.0, max_value=500.0, value=150.0, step=25.0)
    cost_per_km = st.number_input("Travel cost per km", min_value=0.0, value=18.0, step=1.0)
    daily_team_cost = st.number_input("Daily team cost", min_value=0.0, value=3000.0, step=500.0)

    st.header("Distance Engine")
    provider = st.selectbox("Provider", ["Haversine", "OSRM", "openrouteservice"])
    osrm_base_url = st.text_input("OSRM base URL", value="https://router.project-osrm.org")
    ors_api_key = st.text_input("openrouteservice API key", type="password")


if uploaded_file:
    try:
        if uploaded_file.name.lower().endswith(".csv"):
            raw_df = pd.read_csv(uploaded_file)
        else:
            raw_df = pd.read_excel(uploaded_file)

        coordinates_df = clean_input(raw_df)

        effective_walls_per_day = max(
            0.25,
            max_walls_per_team_per_day * weather_productivity_percent / 100 * wall_availability_percent / 100,
        )
        estimated_days_per_branch = math.ceil(walls_per_branch / effective_walls_per_day)
        minimum_teams_for_execution = math.ceil(len(coordinates_df) * estimated_days_per_branch / max_project_days)
        effective_number_of_clusters = min(len(coordinates_df), max(number_of_clusters, number_of_teams))

        if len(coordinates_df) < effective_number_of_clusters:
            st.error("Number of clusters cannot exceed the number of valid branches.")
            st.stop()

        if provider == "openrouteservice" and not ors_api_key:
            st.error("Please enter an openrouteservice API key or switch the provider to Haversine/OSRM.")
            st.stop()

        if minimum_teams_for_execution > number_of_teams:
            st.warning(
                f"With the current productivity assumptions, at least {minimum_teams_for_execution} teams "
                f"are estimated to finish branch execution within {max_project_days} days before travel buffers."
            )

        if effective_number_of_clusters != number_of_clusters:
            st.info(
                f"Using {effective_number_of_clusters} clusters so work can be distributed across {number_of_teams} teams."
            )

        start = time.time()
        progress_bar = st.progress(0, text="Reading coordinates")

        clustered_df, centers_df = cluster_branches(coordinates_df, effective_number_of_clusters)
        progress_bar.progress(20, text="Creating clusters")

        allocated_df = allocate_teams(clustered_df, number_of_teams, estimated_days_per_branch)
        progress_bar.progress(35, text="Allocating teams")

        route_df, travel_summary_df, distance_matrix_df = create_routes(allocated_df, provider, osrm_base_url, ors_api_key)
        progress_bar.progress(60, text="Optimizing routes")

        calendar_df = create_daily_calendar(
            route_df,
            deployment_start,
            walls_per_branch,
            max_walls_per_team_per_day,
            weather_productivity_percent / 100,
            wall_availability_percent / 100,
            max_daily_travel_km,
        )
        progress_bar.progress(75, text="Building daily calendar")

        costing_df = create_costing(travel_summary_df, cost_per_km, daily_team_cost, calendar_df)
        costing_df["deadline_days"] = max_project_days
        costing_df["deadline_variance_days"] = max_project_days - costing_df["working_days"]
        costing_df["deadline_status"] = np.where(
            costing_df["working_days"] <= max_project_days,
            "Within deadline",
            "Exceeds deadline",
        )
        project_days_required = int(costing_df["working_days"].max()) if not costing_df.empty else 0
        total_team_days = int(costing_df["working_days"].sum()) if not costing_df.empty else 0
        minimum_teams_after_travel = math.ceil(total_team_days / max_project_days) if max_project_days else number_of_teams
        deadline_status = "Within deadline" if project_days_required <= max_project_days else "Exceeds deadline"
        dashboard_df = create_dashboard_df(
            coordinates_df,
            allocated_df,
            travel_summary_df,
            costing_df,
            max_project_days,
            project_days_required,
            deadline_status,
            minimum_teams_for_execution,
            minimum_teams_after_travel,
        )
        assumptions_df = pd.DataFrame(
            [
                {"assumption": "Max project days", "value": max_project_days},
                {"assumption": "Walls per branch", "value": walls_per_branch},
                {"assumption": "Max walls per team per day", "value": max_walls_per_team_per_day},
                {"assumption": "Weather/productivity factor", "value": f"{weather_productivity_percent}%"},
                {"assumption": "Wall availability factor", "value": f"{wall_availability_percent}%"},
                {"assumption": "Effective walls per team per day", "value": round(effective_walls_per_day, 2)},
                {"assumption": "Estimated days per branch", "value": estimated_days_per_branch},
                {"assumption": "Max branch visits per team per day", "value": 1},
                {"assumption": "Max practical travel km per day", "value": max_daily_travel_km},
                {"assumption": "Minimum teams for execution deadline", "value": minimum_teams_for_execution},
                {"assumption": "Minimum teams after travel buffers", "value": minimum_teams_after_travel},
                {"assumption": "Project days required", "value": project_days_required},
                {"assumption": "Deadline status", "value": deadline_status},
                {"assumption": "Travel cost per km", "value": cost_per_km},
                {"assumption": "Daily team cost", "value": daily_team_cost},
            ]
        )
        elapsed = round(time.time() - start, 1)
        progress_bar.progress(100, text="Workbook ready")

        st.success(f"Deployment plan generated for {len(coordinates_df)} branches in {elapsed} seconds.")
        if project_days_required <= max_project_days:
            st.success(f"Deadline check: {project_days_required} days required against a {max_project_days}-day deadline.")
        else:
            st.error(
                f"Deadline check: {project_days_required} days required against a {max_project_days}-day deadline. "
                f"Increase teams to at least {minimum_teams_after_travel} or improve productivity assumptions."
            )

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Branches", len(coordinates_df))
        c2.metric("Clusters", effective_number_of_clusters)
        c3.metric("Teams", number_of_teams)
        c4.metric("Project Days", project_days_required)
        c5.metric("Estimated Cost", f"{costing_df['total_cost'].sum():,.0f}")

        dashboard_tab, cluster_tab, route_tab, calendar_tab, costing_tab = st.tabs(
            ["Dashboard", "Clusters", "Routes", "Daily Calendar", "Costing"]
        )

        with dashboard_tab:
            st.subheader("Dashboard")
            st.dataframe(dashboard_df, use_container_width=True)
            st.subheader("Planning Assumptions")
            st.dataframe(assumptions_df, use_container_width=True)

        with cluster_tab:
            st.subheader("Cluster Visualization")
            render_cluster_map(allocated_df)
            st.dataframe(allocated_df, use_container_width=True)

        with route_tab:
            st.subheader("Interactive Route Map")
            render_route_map(route_df)
            st.subheader("Travel Summary")
            st.dataframe(travel_summary_df, use_container_width=True)
            st.subheader("Distance Matrix")
            st.dataframe(distance_matrix_df, use_container_width=True)

        with calendar_tab:
            st.subheader("Daily Calendar")
            st.dataframe(calendar_df, use_container_width=True)

        with costing_tab:
            st.subheader("Costing")
            st.dataframe(costing_df, use_container_width=True)

        workbook = export_workbook(
            {
                "Coordinates": coordinates_df,
                "Clusters": allocated_df,
                "Distance Matrix": distance_matrix_df,
                "Team Allocation": allocated_df,
                "Route Sequence": route_df,
                "Daily Calendar": calendar_df,
                "Travel Summary": travel_summary_df,
                "Costing": costing_df,
                "Dashboard": dashboard_df,
                "Cluster Centers": centers_df,
                "Assumptions": assumptions_df,
            }
        )

        st.download_button(
            "Download deployment workbook",
            data=workbook,
            file_name="au_bank_branch_deployment_plan.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    except Exception as exc:
        st.error(str(exc))
else:
    st.info("Upload a file with at least: branch_name, latitude, longitude. Optional: city, state, pincode.")
