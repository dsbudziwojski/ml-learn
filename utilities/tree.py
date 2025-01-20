class Node():
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

    def left_child(self):
        return self.left

    def right_child(self):
        return self.right

    def __str__(self):
        return str(f'Value: {self.value}')

