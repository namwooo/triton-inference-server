import argparse
import sys
from urllib.request import Request, urlopen
import numpy as np
import cv2

import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException

class_map = {
    1: "sedan_vehicle",
    2: "sedan_vehicle_whole",
    3: "suv_vehicle",
    4: "suv_vehicle_whole",
    5: "van_vehicle",
    6: "van_vehicle_whole",
    7: "bus_vehicle",
    8: "bus_vehicle_whole",
    9: "truck_vehicle",
    10: "truck_vehicle_whole",
    11: "heavyequipment_vehicle",
    12: "heavyequipment_vehicle_whole",
    13: "wheel",
    14: "plate",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-name", type=str, required=True, help="Name of model")
    parser.add_argument(
        "-x",
        "--model-version",
        type=str,
        required=False,
        default="",
        help="Version of model. Default is to user latest version",
    )
    parser.add_argument("-u", "--url", type=str, required=False, default="localhost:8000", help="Model Name")
    parser.add_argument("--image-filename", type=str, nargs="?", default=None, help="Input image / Input folder.")
    return parser.parse_args()


def generate_request(image_data, input_name, outputs, dtype):
    client = httpclient

    input = client.InferInput(input_name, image_data.shape, dtype)
    input.set_data_from_numpy(image_data)

    outputs_ = []
    for output in outputs:
        output_name = output["name"]
        outputs_.append(client.InferRequestedOutput(output_name))

    return input, outputs_, args.model_name, args.model_version


def postprocess(width, height, raw_result):
    result = {
        "width": width,
        "height": height,
        "boxes": raw_result.as_numpy("detection_boxes").squeeze().tolist(),
        "scores": raw_result.as_numpy("detection_scores").squeeze().tolist(),
        "classes": list(map(lambda x: class_map[int(x)], raw_result.as_numpy("detection_classes").squeeze())),
    }
    return result


def get_bboxes(raw_result, threshold):
    boxes = raw_result["boxes"]
    scores = raw_result["scores"]
    classes = raw_result["classes"]
    img_width = raw_result["width"]
    img_height = raw_result["height"]
    pred_bboxes = []
    for index, score in enumerate(scores):
        cls_name = classes[index]
        if score >= threshold:
            box = boxes[index]
            left, right = map(lambda x: x * img_width, (box[1], box[3]))
            top, bottom = map(lambda x: x * img_height, (box[0], box[2]))
            bbox = [cls_name, score, left, top, right, bottom]
            pred_bboxes.append(bbox)

    return pred_bboxes


def get_aimmo_results(bbox_result):
    json_result = {}
    json_result["attributes"] = {}

    annos = []
    for anno_id, bbox in enumerate(bbox_result):
        anno_id_encoded = str(anno_id).zfill(3).encode()
        left, top, right, bottom = map(lambda x: int(round(x)), bbox[2:])

        anno = {}
        anno["type"] = "bbox"
        anno["points"] = [[left, top], [right, top], [right, bottom], [left, bottom]]
        anno["label"] = bbox[0]
        anno["attributes"] = {"score": bbox[1]}
        annos.append(anno)

    json_result["annotations"] = annos

    return json_result


if __name__ == "__main__":
    args = parse_args()

    try:
        triton_client = httpclient.InferenceServerClient(url=args.url)
    except Exception as e:
        print("client creation failed: " + str(e))
        sys.exit(1)

    try:
        model_metadata = triton_client.get_model_metadata(model_name=args.model_name, model_version=args.model_version)
    except InferenceServerException as e:
        print("failed to retrieve the metadata: " + str(e))
        sys.exit(1)

    try:
        model_config = triton_client.get_model_config(model_name=args.model_name, model_version=args.model_version)
    except InferenceServerException as e:
        print("failed to retrieve the config: " + str(e))
        sys.exit(1)

    input_name = model_metadata["inputs"][0]["name"]
    outputs = model_metadata["outputs"]
    dtype = model_metadata["inputs"][0]["datatype"]

    req = Request("file://" + args.image_filename, headers={"User-Agent": "Mozilla/5.0"})
    resp = urlopen(req)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    img_height, img_width, _ = image.shape
    image_data = image.reshape((1, img_height, img_width, 3)).astype(np.uint8)

    input, outputs_, model_name, model_version = generate_request(image_data, input_name, outputs, dtype)

    try:
        response = triton_client.infer(args.model_name, inputs=[input], model_version=model_version, outputs=outputs_)
    except InferenceServerException as e:
        print("inference failed: " + str(e))
        sys.exit(1)
    result = postprocess(img_height, img_width, response)

    pred_bboxes = get_bboxes(result, 0.5)
    pred_aimmo = get_aimmo_results(pred_bboxes)

