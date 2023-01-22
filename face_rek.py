import scipy.io as sio
import pandas as pd
import os
from examples.reid_inference_mdf import FaceReId
import pickle
import csv
import ast
import json
import numpy as np

CUR_FOLDER_PATH = os.path.dirname(os.path.realpath(__file__))
DEST_FOLDER_NAME = "imdb_processed"

class IMDBProcessor:

    def __init__(self):
        self.face_reid = FaceReId()
        self.representations = list()
        self.actor_name_to_paths = dict()
        self.actor_name_to_cluster = list()
        self.representations_dict = dict()

    def compute_clusters(self, path):
        """
        Compute a mean embedding for every character name
        """
        idx = 0
        prev_actor_name = list(self.actor_name_to_paths.keys())[0]
        embeddings = []
        self.representations_dict = {item[0]: item[1] for item in self.representations}
        for actor_name, mdf_paths in self.actor_name_to_paths.items():
            print("Actor name: {}".format(actor_name))

            for mdf_path in mdf_paths:
                print("MDF Path: {}".format(mdf_path))
                if mdf_path in self.representations_dict:
                    cur_embedding = self.representations_dict[mdf_path]
                    embeddings.append(cur_embedding)

            if embeddings:
                mean_embedding = np.mean(embeddings)
                self.actor_name_to_cluster[actor_name] = mean_embedding
                embeddings = []
            # Collect all the embeddings for the same character name
            idx += 1

            if idx % 100 == 0:
                self.save_data(path=path, data=self.actor_name_to_cluster, mode="pickle_dict")

        return

    def process_face(self, image_file_path, dest_file_path):
        
        embedding, mtcnn_cropped_image, names, mdf_id_all, status = self.face_reid.extract_faces(path_mdf=image_file_path,
                                                                                    result_path_good_resolution_faces=dest_file_path,
                                                                                    plot_cropped_faces=True)
        if status:
            mdf_full_path = "/".join(dest_file_path.split("/")[-2:])
            instance = []
            instance.append(mdf_full_path)
            instance.append(embedding)
            self.representations.append(instance)

        return status

    def process_images(self, path):
        with open(path, "r") as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                if idx > 200000:
                    image_path = row[2]
                    face_conf_1 = row[3]
                    face_conf_2 = row[4]
                    if not face_conf_2 and face_conf_1 != '-inf':
                        print("Index: {}".format(idx))
                        actor_name = ast.literal_eval(row[5])[0]

                        image_full_path = os.path.join(CUR_FOLDER_PATH, image_path)
                        dest_file_path = os.path.join(CUR_FOLDER_PATH, image_path.replace("imdb_crop", DEST_FOLDER_NAME))
                        dest_image_path = "/".join(dest_file_path.split("/")[:-1])
                        os.makedirs(dest_image_path, exist_ok=True)
                        face_found = self.process_face(image_full_path, dest_file_path)
                        if face_found:
                            image_path = "/".join(image_path.split("/")[-2:])
                            if actor_name not in self.actor_name_to_paths:
                                self.actor_name_to_paths[actor_name] = [image_path]
                            else:
                                self.actor_name_to_paths[actor_name].append(image_path)

                    if idx % 10000 == 0:
                        print("Saved: {}".format(idx))
                        path = os.path.join(CUR_FOLDER_PATH, "actor_name_to_paths_part2.json")
                        if self.actor_name_to_paths:        
                            self.save_data(path, data=self.actor_name_to_paths, mode="json")
                        path = os.path.join(CUR_FOLDER_PATH, "representations_part2.pkl")
                        if self.representations:
                            self.save_data(path, data=self.representations, mode="pickle")
                        print("Done.")

    def save_data(self, path, data, mode="pickle"):
        if mode == "pickle":
            with open(path, "wb") as f:
                pickle.dump(data, f)

        elif mode == "pickle_dict":
            with open(path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        elif mode == "json":
            with open(path, "w") as f:
                json.dump(data, f)


    def read_data(self, path, mode="pickle"):
        if mode == "pickle":
            with open(path, "rb") as f:
                data = pickle.load(f)
                return data

        elif mode == "json":
            with open(path, "r") as f:
                data = json.load(f)
                return data


def main():
    imdb_processor = IMDBProcessor()
    imdb_path = "/notebooks/imdb_data.csv"
    # imdb_data = imdb_processor.build_imdb_data(imdb_path)
    # images_path = "/notebooks/nebula3_face_rekognition/imdb_crop"
    # imdb_processor.actor_name_to_paths = imdb_processor.read_data("/notebooks/nebula3_face_rekognition/actor_name_to_paths_copy.json", mode="json")
    # imdb_processor.representations = imdb_processor.read_data("/notebooks/nebula3_face_rekognition/representations.pkl", mode="pickle")
    # path = os.path.join(CUR_FOLDER_PATH, "actor_name_to_cluster.pkl")
    # imdb_processor.compute_clusters(path) 
    imdb_processor.process_images(imdb_path)


if __name__ == '__main__':
    main()










    # def process_images(self, path):
    #     for root, subdirs, files in os.walk(path):
    #         dest_folder = "imdb_processed"
    #         src_folder = root.split("/")[-1]
    #         dest_path = root.replace(src_folder, dest_folder)

    #         for subdir in subdirs:
    #             subdir_root_path = os.path.join(root, subdir)
    #             subdir_dest_path = os.path.join(dest_path, subdir)
    #             os.makedirs(subdir_dest_path, exist_ok=True)

    #             image_files = [os.path.join(subdir_root_path, image_file) for image_file in os.listdir(subdir_root_path) if image_file.endswith(".jpg")]
    #             image_dest_files = [os.path.join(subdir_dest_path, image_file) for image_file in os.listdir(subdir_root_path) if image_file.endswith(".jpg")]
                
    #             for idx, image_file_path in enumerate(image_files):
    #                 image_dest_path = "/".join(image_dest_files[idx].split("/")[:-1])
    #                 os.makedirs(image_dest_path, exist_ok=True)
    #                 self.process_face(image_file_path, image_dest_path) 

    #                 # Dump all the data to Pickle file
    #                 self.save_data()     
                    
            # self.save_data()  