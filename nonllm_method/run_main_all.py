import os
import json
import subprocess

# Define data dictionaries
data_dicts = [
    {
        "data_name": "bank",
        "nodes": ["IG01", "IG02", "Tomcat01", "Tomcat02", "Tomcat03", "Tomcat04", "MG01", "MG02"],
        "dependency_graph": {
            "apache01": ["IG01", "IG02"],
            "apache02": ["IG01", "IG02"],
            "IG01": ["Tomcat01", "Tomcat02", "Tomcat03", "Tomcat04"],
            "IG02": ["Tomcat01", "Tomcat02", "Tomcat03", "Tomcat04"],
            "Tomcat01": ["Redis01", "Mysql01", "MG01", "MG02"],
            "Tomcat02": ["Redis01", "Mysql01", "MG01", "MG02"],
            "Tomcat03": ["Redis01", "Mysql01", "MG01", "MG02"],
            "Tomcat04": ["Redis01", "Mysql01", "MG01", "MG02"],
            "MG01": ["Redis02", "Mysql02"],
            "MG02": ["Redis02", "Mysql02"],
        },
        "jump_folder": [
            0,
            1,
            6,
            13,
            15,
            23,
            24,
            25,
            27,
            28,
            35,
            37,
            57,
            60,
            64,
            68,
            69,
            70,
            73,
            80,
            86,
            89,
            90,
            93,
            110,
            119,
            120,
            126,
            128,
            129,
            132,
            134,
            135,
        ],
    },
    {
        "data_name": "telecom",
        "nodes": [
            "docker_001",
            "docker_002",
            "docker_003",
            "docker_004",
            "docker_005",
            "docker_006",
            "docker_007",
            "docker_008",
        ],
        # According to all components in Telecom dataset, provide complete dependency graph where each node depends on all other nodes except itself
        "dependency_graph": {
            node: [
                n
                for n in [
                    "docker_001",
                    "docker_002",
                    "docker_003",
                    "docker_004",
                    "docker_005",
                    "docker_006",
                    "docker_007",
                    "docker_008",
                ]
                if n != node
            ]
            for node in [
                "docker_001",
                "docker_002",
                "docker_003",
                "docker_004",
                "docker_005",
                "docker_006",
                "docker_007",
                "docker_008",
            ]
        },
        "jump_folder": [
            2,
            5,
            6,
            8,
            11,
            12,
            13,
            14,
            16,
            19,
            20,
            21,
            22,
            24,
            25,
            26,
            27,
            29,
            30,
            32,
            33,
            34,
            36,
            38,
            39,
            42,
            43,
            44,
            46,
            47,
            48,
            49,
        ],
    },
    {
        "data_name": "market1",
        "nodes": [
            "frontend-0",
            "frontend-1",
            "frontend-2",
            "frontend2-0",
            "shippingservice-0",
            "shippingservice-1",
            "shippingservice-2",
            "shippingservice2-0",
            "checkoutservice-0",
            "checkoutservice-1",
            "checkoutservice-2",
            "checkoutservice2-0",
            "currencyservice-0",
            "currencyservice-1",
            "currencyservice-2",
            "currencyservice2-0",
            "adservice-0",
            "adservice-1",
            "adservice-2",
            "adservice2-0",
            "emailservice-0",
            "emailservice-1",
            "emailservice-2",
            "emailservice2-0",
            "cartservice-0",
            "cartservice-1",
            "cartservice-2",
            "cartservice2-0",
            "productcatalogservice-0",
            "productcatalogservice-1",
            "productcatalogservice-2",
            "productcatalogservice2-0",
            "recommendationservice-0",
            "recommendationservice-1",
            "recommendationservice-2",
            "recommendationservice2-0",
            "paymentservice-0",
            "paymentservice-1",
            "paymentservice-2",
            "paymentservice2-0",
        ],
        "dependency_graph": {
            # checkoutservice -> emailservice, cartservice, shippingservice, currencyservice, paymentservice, productcatalogservice
            "checkoutservice-0": [
                "emailservice-0",
                "cartservice-0",
                "shippingservice-0",
                "currencyservice-0",
                "paymentservice-0",
                "productcatalogservice-0",
            ],
            "checkoutservice-1": [
                "emailservice-1",
                "cartservice-1",
                "shippingservice-1",
                "currencyservice-1",
                "paymentservice-1",
                "productcatalogservice-1",
            ],
            "checkoutservice-2": [
                "emailservice-2",
                "cartservice-2",
                "shippingservice-2",
                "currencyservice-2",
                "paymentservice-2",
                "productcatalogservice-2",
            ],
            "checkoutservice2-0": [
                "emailservice2-0",
                "cartservice2-0",
                "shippingservice2-0",
                "currencyservice2-0",
                "paymentservice2-0",
                "productcatalogservice2-0",
            ],
            # frontend -> productcatalogservice, cartservice, currencyservice, adservice, shippingservice, checkoutservice, recommendationservice
            "frontend-0": [
                "productcatalogservice-0",
                "cartservice-0",
                "currencyservice-0",
                "adservice-0",
                "shippingservice-0",
                "checkoutservice-0",
                "recommendationservice-0",
            ],
            "frontend-1": [
                "productcatalogservice-1",
                "cartservice-1",
                "currencyservice-1",
                "adservice-1",
                "shippingservice-1",
                "checkoutservice-1",
                "recommendationservice-1",
            ],
            "frontend-2": [
                "productcatalogservice-2",
                "cartservice-2",
                "currencyservice-2",
                "adservice-2",
                "shippingservice-2",
                "checkoutservice-2",
                "recommendationservice-2",
            ],
            "frontend2-0": [
                "productcatalogservice2-0",
                "cartservice2-0",
                "currencyservice2-0",
                "adservice2-0",
                "shippingservice2-0",
                "checkoutservice2-0",
                "recommendationservice2-0",
            ],
            # recommendationservice -> productcatalogservice
            "recommendationservice-0": ["productcatalogservice-0"],
            "recommendationservice-1": ["productcatalogservice-1"],
            "recommendationservice-2": ["productcatalogservice-2"],
            "recommendationservice2-0": ["productcatalogservice2-0"],
        },
        # "jump_folder": [2, 3, 6, 8, 9, 10, 14, 15, 19, 23, 24, 36, 37, 38, 40, 43, 46, 50, 51, 52, 61, 62, 63, 64, 65, 66, 1, 4, 5, 7, 11, 12, 13, 16, 17, 18, 20, 21, 22, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 39, 41, 42, 44],
        "jump_folder": [
            2,
            3,
            6,
            8,
            9,
            10,
            14,
            15,
            19,
            23,
            24,
            36,
            37,
            38,
            40,
            43,
            46,
            50,
            51,
            52,
            61,
            62,
            63,
            64,
            65,
            66,
        ],
    },
    {
        "data_name": "market2",
        "nodes": [
            "frontend-0",
            "frontend-1",
            "frontend-2",
            "frontend2-0",
            "shippingservice-0",
            "shippingservice-1",
            "shippingservice-2",
            "shippingservice2-0",
            "checkoutservice-0",
            "checkoutservice-1",
            "checkoutservice-2",
            "checkoutservice2-0",
            "currencyservice-0",
            "currencyservice-1",
            "currencyservice-2",
            "currencyservice2-0",
            "adservice-0",
            "adservice-1",
            "adservice-2",
            "adservice2-0",
            "emailservice-0",
            "emailservice-1",
            "emailservice-2",
            "emailservice2-0",
            "cartservice-0",
            "cartservice-1",
            "cartservice-2",
            "cartservice2-0",
            "productcatalogservice-0",
            "productcatalogservice-1",
            "productcatalogservice-2",
            "productcatalogservice2-0",
            "recommendationservice-0",
            "recommendationservice-1",
            "recommendationservice-2",
            "recommendationservice2-0",
            "paymentservice-0",
            "paymentservice-1",
            "paymentservice-2",
            "paymentservice2-0",
        ],
        "dependency_graph": {
            # checkoutservice -> emailservice, cartservice, shippingservice, currencyservice, paymentservice, productcatalogservice
            "checkoutservice-0": [
                "emailservice-0",
                "cartservice-0",
                "shippingservice-0",
                "currencyservice-0",
                "paymentservice-0",
                "productcatalogservice-0",
            ],
            "checkoutservice-1": [
                "emailservice-1",
                "cartservice-1",
                "shippingservice-1",
                "currencyservice-1",
                "paymentservice-1",
                "productcatalogservice-1",
            ],
            "checkoutservice-2": [
                "emailservice-2",
                "cartservice-2",
                "shippingservice-2",
                "currencyservice-2",
                "paymentservice-2",
                "productcatalogservice-2",
            ],
            "checkoutservice2-0": [
                "emailservice2-0",
                "cartservice2-0",
                "shippingservice2-0",
                "currencyservice2-0",
                "paymentservice2-0",
                "productcatalogservice2-0",
            ],
            # frontend -> productcatalogservice, cartservice, currencyservice, adservice, shippingservice, checkoutservice, recommendationservice
            "frontend-0": [
                "productcatalogservice-0",
                "cartservice-0",
                "currencyservice-0",
                "adservice-0",
                "shippingservice-0",
                "checkoutservice-0",
                "recommendationservice-0",
            ],
            "frontend-1": [
                "productcatalogservice-1",
                "cartservice-1",
                "currencyservice-1",
                "adservice-1",
                "shippingservice-1",
                "checkoutservice-1",
                "recommendationservice-1",
            ],
            "frontend-2": [
                "productcatalogservice-2",
                "cartservice-2",
                "currencyservice-2",
                "adservice-2",
                "shippingservice-2",
                "checkoutservice-2",
                "recommendationservice-2",
            ],
            "frontend2-0": [
                "productcatalogservice2-0",
                "cartservice2-0",
                "currencyservice2-0",
                "adservice2-0",
                "shippingservice2-0",
                "checkoutservice2-0",
                "recommendationservice2-0",
            ],
            # recommendationservice -> productcatalogservice
            "recommendationservice-0": ["productcatalogservice-0"],
            "recommendationservice-1": ["productcatalogservice-1"],
            "recommendationservice-2": ["productcatalogservice-2"],
            "recommendationservice2-0": ["productcatalogservice2-0"],
        },
        "jump_folder": [
            1,
            5,
            11,
            12,
            17,
            18,
            20,
            21,
            24,
            25,
            28,
            32,
            36,
            38,
            42,
            44,
            46,
            48,
            52,
            53,
            61,
            64,
            68,
            69,
            71,
        ],
    },
]

best_result_data_dict = {
    "nezha": {
        "bank": ["8_20210325_040500"],
        "telecom": ["4_20200525_003500"],
        "market1": [],
        "market2": [],
    },
    "pdiagnose": {
        "bank": [],
        "telecom": [],
        "market1": [],
        "market2": [],
    },
    "microrca": {
        "bank": ["0_20210304_040500"],
        "telecom": ["1_20200522_010700"],
        "market1": [],
        "market2": [],
    },
    "microscope": {
        "bank": ["2_20210312_163500"],
        "telecom": ["7_20200528_000700"],
        "market1": [],
        "market2": [],
    },
    "microrank": {
        "bank": ["0_20210304_040500"],
        "telecom": ["6_20200527_033500"],
        "market1": [],
        "market2": [],
    },
    "traceanomaly": {
        "bank": ["3_20210309_040500"],
        "telecom": ["0_20200411_003500"],
        "market1": [],
        "market2": [],
    },
}


def run_main_all():

    # Algorithm list
    algorithms = ["nezha", "pdiagnose", "microrank", "microrca", "microscope", "traceanomaly"]


    # Iterate through data dictionary list
    for data_item in data_dicts:
        data_name = data_item["data_name"]
        print(f"Starting to process dataset: {data_name}", flush=True)
        # Iterate through algorithms
        for algorithm in algorithms:
            print(
                f"========== Data: {data_name}, Algorithm: {algorithm}, Starting processing ==========", flush=True
            )
            # Build base path

            base_path = f"/data/dds-data/opslens/{data_name}/compare_data/00"
            # Get subfolder list and sort by timestamp
            all_items = os.listdir(base_path)

            subfolders = [
                item for item in all_items if os.path.isdir(os.path.join(base_path, item))
            ]

            subfolders.sort(key=lambda x: int(x.split("_")[0]))
            print(f"subfolders: {subfolders}")

            # Get subfolders from best_result_data_dict based on data_name and algorithm
            best_subfolders = best_result_data_dict.get(algorithm, {}).get(data_name, [])
            if best_subfolders:
                subfolders = [sf for sf in best_subfolders if sf in subfolders]
                if not subfolders:
                    subfolders = subfolders[:1]
            else:
                subfolders = subfolders[:1]

            subfolders = subfolders[:1]

            # Iterate through sorted subfolders
            for subfolder in subfolders:

                print(f"Starting to process subfolder: {subfolder}", flush=True)

                # Check if output directory already exists
                output_dir = f"/home/dds/ljw/paper/opslens/autogen04/data_convert/baseline/results_new/{algorithm}/{data_name}/{subfolder}"


                subfolder_path = os.path.join(base_path, subfolder)
                algorithm_config = {
                    "data_name": data_item["data_name"],
                    "nodes": data_item["nodes"],
                    "dependency_graph": data_item["dependency_graph"],
                    "jump_folder": data_item["jump_folder"],
                    "normal_data_dir": subfolder_path,
                    "data_final_dir": f"/data/dds-data/opslens/{data_name}/compare_data",
                    "output_dir": output_dir,
                    "failure_label_path": f"/home/dds/ljw/paper/opslens/autogen04/data_convert/baseline/failure_label_{data_name}.json",
                }

                algorithm_config_path = (
                    f"/home/dds/ljw/paper/opslens/autogen04/data_convert/baseline/config.json"
                )
                # Write configuration file to config.json
                print(f"Writing configuration file to: {algorithm_config_path}", flush=True)
                with open(algorithm_config_path, "w") as f:
                    json.dump(algorithm_config, f, indent=2)

                # Ensure output directory exists
                print(f"Creating output directory: {algorithm_config['output_dir']}", flush=True)
                os.makedirs(algorithm_config["output_dir"], exist_ok=True)
                # Write configuration file to output_dir/config.json
                with open(f"{algorithm_config['output_dir']}/config.json", "w") as f:
                    json.dump(algorithm_config, f, indent=2)

                # Fix subprocess call
                log_file = f"{algorithm_config['output_dir']}/run.log"
                python_file = f"{algorithm}_parallel.py"
                cmd = f"python {python_file} > {log_file}"
                print(f"Processing, Data: {data_name}, Algorithm: {algorithm}, Normal data: {subfolder_path}")
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                print(f"Command execution completed, returncode: {result.returncode}", flush=True)
                print(f"stdout: {result.stdout}", flush=True)
                print(f"stderr: {result.stderr}", flush=True)

            print(f"Processing completed, Data: {data_name}, Algorithm: {algorithm}")
        print("-" * 50)  # Separator line


if __name__ == "__main__":
    run_main_all()
