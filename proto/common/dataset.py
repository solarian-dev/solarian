from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
# import threading
import multiprocessing
import time
import pickle
import os

multiprocessing.set_start_method('fork')


class Dataset:


    def __init__(self, repo_id, fps, root, robot_type, features, use_videos, image_writer_processes, image_writer_threads):

        self.repo_id = repo_id
        self.fps = fps
        self.root = root
        self.robot_type = robot_type
        self.features = features
        self.use_videos = use_videos
        self.image_writer_processes = image_writer_processes
        self.image_writer_threads = image_writer_threads
        
        self.episode = []
        self.data = []
        self.episode_number = 0
        self.incremental_store = True

        self.dataset = LeRobotDataset.create(
            self.repo_id,
            self.fps,
            root=self.root,
            robot_type=self.robot_type,
            features=self.features,
            use_videos=self.use_videos,
            image_writer_processes=self.image_writer_processes,
            image_writer_threads=self.image_writer_threads,
        )

        # create pickles folder
        os.makedirs(f"{self.root}/pickles", exist_ok=True)

        with open(f"{self.root}/pickles/features.pkl", "wb") as f:
            pickle.dump(self.features, f)

        if self.incremental_store:
            self.storing = multiprocessing.Manager().Event()
            self.storing.clear()


    def add_frame(self, frame, task):
        self.episode.append({"frame": frame, "task": task})
        

    def save_episode(self):
        # self.data.append(self.episode)
        

        # save episode as pickle file using dataset name as folder and episode number as filename
        with open(f"{self.root}/pickles/episode_{self.episode_number}.pkl", "wb") as f:
            pickle.dump(self.episode, f)
        
        self.episode = []
        self.episode_number += 1
        # if self.incremental_store and not self.storing.is_set():
        #     episode = self.data.pop(0)
        #     multiprocessing.Process(target=store_oldest_episode, args=(self.storing, self.dataset, episode)).start()
        

    def clear_episode_buffer(self):
        self.episode = []



    def store_dataset(self):
        # stores dataset as lerobot dataset
        
        for episode in self.data:
            for frame in episode:
                self.dataset.add_frame(frame["frame"], frame["task"])
            self.dataset.save_episode()

    def push_to_hub(self, tags=None, private=False):
        self.dataset.push_to_hub(tags=tags, private=private)
        


def store_oldest_episode(storing, dataset, episode):
    storing.set()
    # time.sleep(1)
    print("Storing episode")
    for frame in episode:
        dataset.add_frame(frame["frame"], frame["task"])
    dataset.save_episode()
    storing.clear()