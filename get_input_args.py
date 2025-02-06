import argparse

# python train.py data_dir --save_dir save_directory


def get_input_args_train():
    """
    Train model command Line Arguments:
      1. dir to save checkpoint --save_dir with default value 'save_directory'
      2. Choose Architecture as --arch with default value 'vgg11' or 'vgg13'
      3. set hyperparameters:Learning rate --learning_rate with default value '0.001'
      4. set hyperparameters:hidden units --hidden_units with default value '512'
      5. set hyperparameters:epochs --epochs with default value 20
      6. GPU --gpu (otherwise use cpu)
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    # python train.py "flowers" --save_dir "save_model" --arch "vgg16" --learning_rate 0.01 --hidden_units 512 --epochs 6
    # Create Parse using ArgumentParser
    # model is the modal name for eg. vgg ,resnet, alexnet
    parser = argparse.ArgumentParser(description='arguments to train model')

    # position argument
    parser.add_argument('data_dir', type=str,
                        help='Path to the data directory')
    # the following are all options arg
    parser.add_argument('--save_dir', type=str,
                        default='save_directory', help='dir to save checkpoint')
    parser.add_argument('--arch', type=str, default='vgg16',
                        help='need argument --arch')
    parser.add_argument('--learning_rate', type=float,
                        default=0.001, help='argument for learning rate')
    parser.add_argument('--hidden_units', type=int,
                        default=512, help='argument for model hidden layer')
    parser.add_argument('--epochs', type=int,
                        default=10, help='argument for epochs to train the model')
    parser.add_argument('--gpu', action='store_true',
                        help='argument to use gpu if supports')

    # Replace None with parser.parse_args() parsed argument collection that
    # you created with this function
    # data structure that stores the command line arguments object
    args = parser.parse_args()
    return args


def get_input_args_predict():
    """
     Predict command Line Arguments:
     basic usage python predict.py /path/to/image checkpoint
      1. image_dir - image path for the image to image classification
      2. checkpoint - saved checkpoint for the model
     Options:
     1. --top_k return topK most likely classes:
     2. --category_names map categories to real names:
     3. GPU --gpu (otherwise use cpu)
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  

     for eg. python predict.py 'flowers/train/1/image_06734.jpg' 'checkpoint.pth' --top_k 5 --category_names 'cat_to_name.json' --gpu
    """
    # Create Parse using ArgumentParser
    # model is the modal name for eg. vgg ,resnet, alexnet
    parser = argparse.ArgumentParser(description='to pass 3 arguments.')
    # image path
    parser.add_argument('image_dir', type=str,
                        help='Path to the image directory')
    # model checkpoint
    parser.add_argument('checkpoint', type=str,
                        help='model checkpoint',  default='checkpoint.pth')
    # the following are all options arg
    parser.add_argument('--top_k', type=str,
                        default='5', help='specify the topK most likely classes')
    parser.add_argument('--category_names', type=str,
                        default='cat_to_name.json', help='the file that have the category name mapping to plot the graph')
    parser.add_argument('--gpu', action='store_true',
                        help='argument to use gpu if supports')

    args = parser.parse_args()
    return args


def check_training_command_line_arguments(in_arg):
    """
    For Lab: Classifying Images - 7. Command Line Arguments
    Prints each of the command line arguments passed in as parameter in_arg, 
    assumes you defined all three command line arguments as outlined in 
    '7. Command Line Arguments'
    Parameters:
     in_arg -data structure that stores the command line arguments object
    Returns:
     Nothing - just prints to console  
    """
    if in_arg is None:
        print("* Doesn't Check the Command Line Arguments because 'get_input_args_predict' hasn't been defined.")
    if (in_arg.learning_rate == 0 or in_arg.learning_rate > 0.1):
        raise ValueError("learning_rate must be between  0 and 0.1")
    if (in_arg.epochs == 0 or in_arg.epochs > 30):
        raise ValueError("Epochs must be between 1 and 30.")
    # check hidden unit should not be 0 or too large
    if (in_arg.hidden_units <= 0 or in_arg.hidden_units > 2046):
        raise ValueError(
            "hidden_units must be between 1 and 2046. Tensor creation does not allow negative dimension")

    if (in_arg.save_dir is None or in_arg.save_dir == ""):
        in_arg.save_dir = "save_model"

    else:
        # prints command line args
        print("Command Line Arguments:\n data_dir =", in_arg.data_dir,
              "\n save_dir =", in_arg.save_dir,
              "\n arch =", in_arg.arch,
              "\n learning_rate =", in_arg.learning_rate,
              "\n hidden_units =", in_arg.hidden_units,
              "\n epochs =", in_arg.epochs,
              "\n gpu =", in_arg.gpu)


def check_predict_command_line_arguments(in_arg):
    """
    For Lab: Classifying Images - 7. Command Line Arguments
    Prints each of the command line arguments passed in as parameter in_arg, 
    assumes you defined all three command line arguments as outlined in 
    '7. Command Line Arguments'
    Parameters:
     in_arg -data structure that stores the command line arguments object
    Returns:
     Nothing - just prints to console  
    """
    if in_arg is None:
        print("* Doesn't Check the Command Line Arguments because 'get_input_args_train' hasn't been defined.")
    else:
        # Validate and process top_k
        try:
            top_k = int(in_arg.top_k)
            if top_k > 10 or top_k == 0:
                raise ValueError(
                    "Invalid value for top_k. It must be between 1 and 10.")
        except ValueError:
            print("Invalid value for top_k. It must be an integer.")

        # Print command line arguments
        print("Command Line Arguments:")
        print(" image_dir =", in_arg.image_dir)
        print(" checkpoint =", in_arg.checkpoint)
        print(" top_k =", in_arg.top_k)
        print(" category_names =", in_arg.category_names)
        print(" gpu =", in_arg.gpu)
