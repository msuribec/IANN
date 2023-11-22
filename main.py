from src.utils import CreateVenv


if __name__ == "__main__":
    c = CreateVenv() #create virtual environment
    c.install_requirements('requirements.txt') #install packages from requirements file
    c.run_file('src/perceptron.py') #run file with list of arguments

