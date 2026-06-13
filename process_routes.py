from retrosynformer.data import RouteDataset
import argparse


def create_route_pickle(
    route_path,
    building_block_path,
    template_library,
    templates_path,
    save_routes_pickle_path,
):
    route_dataset = RouteDataset(
        routes_path=route_path,
        building_block_path=building_block_path,
        templates_path=templates_path,
        template_library=template_library,
    )

    data = route_dataset.create_route_data()
    print(data)
    data.to_json(save_routes_pickle_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Program level arguments
    parser.add_argument(
        "-r",
        "--route_path",
        type=str,
        help="The path and filename of the file with routes to use",
    )
    parser.add_argument(
        "-bb",
        "--building_block_path",
        type=str,
        help="The path and filename of the building block file to use",
    )
    parser.add_argument(
        "-tl",
        "--template_library_path",
        type=str,
        help="The path and filename of the template library file to use",
    )
    parser.add_argument(
        "-t",
        "--templates_path",
        type=str,
        help="The path and filename of the templates file to use",
    )
    parser.add_argument(
        "-s",
        "--save_path",
        type=str,
        help="The path and filename where to save data as pickle.",
    )

    args = parser.parse_args()

    create_route_pickle(
        route_path=args.route_path,
        building_block_path=args.building_block_path,
        template_library=args.template_library_path,
        templates_path=args.templates_path,
        save_routes_pickle_path=args.save_path,
    )
