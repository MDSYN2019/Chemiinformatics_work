"""
Define abstract classes for other classes to use here. We will
define the neural network class to build upon.

We may also want to create abstract properties and force our
subclass to implement those properties. This could be done
by @property decorator along with @abstractmethod
"""


from abc import ABC, abstractmethod

"""
getters and setters are methods that are used to access and
modify the values of private instances of a class. Private instance variables
are variables 

Getters and setters are used to provide controlled access to these
private instance variables. Getters are methods that allow you
to retrieve the current value of a private instance variable, while
setters are methods that allow you to set a new value for a p
"""


class MyClass:
    def __init__(self):
        self._my_variable = None

    def get_my_variable(self):
        return self.get_my_variable

    def set_my_variable(self, new_value):
        self._my_variable = new_value


class molecular_modelling_neural_network(ABC):
    """
    abstract method for the main neural network training model
    which to inherit from and to initate the methods necessary

    what methods does this support?

    NNParameters - initate all outside inputs and store in the class


    """

    @property
    def return_NN_parameters(self):
        """
        ensure we have a number of hidden layers larger than 0
        """
        return self.n_hidden_layers

    @abstractmethod
    def prepare_model(self) -> None:
        """
        this step prepares the inner network and fits the training data into
        the model
        """
        pass

    @abstractmethod
    def _compile_model(self) -> None:
        """ """
        pass


class Animal(ABC):
    @property
    def food_eaten(self):
        return self._food

    """
    Since animals often have different diets, we'll need to define
    a diet in our animals classes. Since all animals 
    Since animals often have different diets, we'll need to define 
    a diet in our animal classes. Since all animals are inheriting 
    look up getters and setters 
    """

    @food_eaten.setter
    def food_eaten(self, food):
        if food in self.diet:  # ensure that the food is appropriate for the animal
            self._food = food
        else:
            raise ValueError(
                f"You can't feed this animal with {food}"
            )  # else throw error

    @property
    @abstractmethod
    def diet(self):
        pass

    @abstractmethod
    def feed(self, time):
        pass
