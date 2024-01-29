import os


# Used to compute hyperparameter training time from SimCSE trained model time.
def compute_training_time():
    sup_batch16_sum = 0.0
    sup_batch32_sum = 0
    unsup_batch16_sum = 0
    unsup_batch32_sum = 0

    path = "model/sup-batch16"
    # path = "test_dir"
    dirs = os.listdir(path)

    # # This would print all the files and directories
    for dir in dirs:
        files = os.listdir(path + "/" + dir)
        # print("files: ", files)
        for file in files:
            if file == "train_results.txt":
                with open(path + "/" + dir + "/" + file, "r") as f:
                    lines = f.readlines()
                    arr = lines[1].split("=")
                    sup_batch16_sum += float(arr[1])

    path = "model/sup-batch32"
    dirs = os.listdir(path)

    # # This would print all the files and directories
    for dir in dirs:
        files = os.listdir(path + "/" + dir)
        for file in files:
            if file == "train_results.txt":
                with open(path + "/" + dir + "/" + file, "r") as f:
                    lines = f.readlines()
                    arr = lines[1].split("=")
                    sup_batch32_sum += float(arr[1])


    path = "model/new-unsup-batch16"
    dirs = os.listdir(path)

    # # This would print all the files and directories
    for dir in dirs:
        files = os.listdir(path + "/" + dir)
        for file in files:
            if file == "train_results.txt":
                with open(path + "/" + dir + "/" + file, "r") as f:
                    lines = f.readlines()
                    arr = lines[1].split("=")
                    unsup_batch16_sum += float(arr[1])


    path = "model/new-unsup-batch32"
    dirs = os.listdir(path)

    # # This would print all the files and directories
    for dir in dirs:
        files = os.listdir(path + "/" + dir)
        for file in files:
            if file == "train_results.txt":
                with open(path + "/" + dir + "/" + file, "r") as f:
                    lines = f.readlines()
                    arr = lines[1].split("=")
                    unsup_batch32_sum += float(arr[1])

    print("Supervised batch 16 training time: ", sup_batch16_sum)
    print("Supervised batch 32 training time: ", sup_batch32_sum)
    print("Unsupervised batch 16 training time: ", unsup_batch16_sum)
    print("Unsupervised batch 32 training time: ", unsup_batch32_sum)

if __name__ == "__main__":
    compute_training_time()