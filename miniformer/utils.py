import os

def train_test_split(filepath, test_size=0.2):
    """
    Splits a text file into training and testing sets based on the specified test size.
    Args:
        filepath (str): Path to the input text file.
        test_size (float, optional): Proportion of the data to include in the test split (default is 0.2).
    Side Effects:
        Creates 'train.txt' and 'test.txt' files in the same directory as the input file, 
        containing the training and testing splits respectively.
    """

    if filepath.endswith('.txt'):

        print("Processing text file:", filepath)
        with open(filepath, 'r') as f:
            raw_text = f.read()

        train_text = raw_text[: int(len(raw_text) * (1 - test_size))]
        test_text = raw_text[int(len(raw_text) * (1 - test_size)):]
        print("Train text length:", len(train_text))
        print("Test text length:", len(test_text))

        output_folder = '/'.join(filepath.split('/')[:-1])
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        with open(os.path.join(output_folder, 'train.txt'), 'w') as f:
            f.write(train_text)
        with open(os.path.join(output_folder, 'test.txt'), 'w') as f:
            f.write(test_text)

    return   

if __name__ == "__main__":

    filepath = "/home/ssubrahmanya/gpt2/miniFormer/data/the-verdict.txt"
    train_test_split(filepath)