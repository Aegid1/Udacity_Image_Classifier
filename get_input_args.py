import argparse
from train import train_model
import predict
from predict import load_model, process_image, predict_with_gpu, predict_without_gpu, show_result

def main():
    
        parser = argparse.ArgumentParser(description = "arguments for trainig the model")
    
        parser.add_argument("--model", action = "store", dest = "model_architecture", help = "decide between vgg13 and densenet18")
        parser.add_argument("--lr", action = "store", dest = "lr", help = "which learnrate u are going to use", type = float)
        parser.add_argument("--epochs", action = "store", dest = "epochs", help = "how many epochs you want to train your model", type = int)
        parser.add_argument("--gpu", action = "store", dest = "gpu", help = "if you want to use gpu or not, enter yes or no")
        parser.add_argument("--h1", action = "store", dest = "h1", help = "defines the amount of hidden_untis in hidden_layer 1", type = int)
        parser.add_argument("--h2", action = "store", dest = "h2", help = "defines the amount of hidden_untis in hidden_layer 2", type = int)
        parser.add_argument("--topk", action = "store", dest = "topk", help = "defines the amount of classes that are displayed after the prediction", type = int)
    
        args = parser.parse_args()
        arguments = []
    
        for arg in vars(args):
        
            arguments.append(getattr(args, arg))
            print(arguments)
#     print(arguments[0])
        checkpoint = input("Do you want to load the existing trained model? Enter yes or no ")
    
        if checkpoint == "yes":
            
            model = load_model("./checkpoint.pth")
            image_file_name = input("Which image do you want to classify, insert the filepath of the image:")
        
            if arguments[3] == "yes":
                probs, classes = predict_with_gpu(image_file_name, model, topk = arguments[6])
            else:
                probs, classes = predict_without_gpu(image_file_name, model, topk = arguments[6])
        
            show_result(probs, classes)
        
        else:
        
            train_model(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6])
            model = load_model("./checkpoint.pth")
    
            image_file_name = input("Which image do you want to classify, insert the filepath of the image:")
        
            if arguments[3] == "yes":
                probs, classes = predict_with_gpu(image_file_name, model, topk = arguments[6])
            else:
                probs, classes = predict_without_gpu(image_file_name, model, topk = arguments[6])
            
            show_result(probs, classes)
    

if __name__ == "__main__":
    main()