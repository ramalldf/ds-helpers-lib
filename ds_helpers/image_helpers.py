from skimage.util import view_as_windows 
from matplotlib import pyplot as plt

class imagePatcher:
  def __init__(self, input_arr, output_dim=(8,8)):
    '''Breaks image into patches and returns them in a single list'''

    # Define step size as equal to the size of the rows
    self.input_arr = input_arr 
    self.output_dim = output_dim 
    self.step_size = output_dim[0]
    self.vmin = min(input_arr.flatten())
    self.vmax = max(input_arr.flatten())

  def patch_image(self):
    '''Breaks apart image into patches'''

    # Patch
    temp = view_as_windows(self.input_arr, self.output_dim, self.step_size)

    self.rows = temp.shape[0]

    # Rearrange patches a single list
    temp_list = [temp_j for temp_i in temp for temp_j in temp_i]

    self.num_patches = len(temp_list)

    return temp_list


  def show_image_patches(self, input_list, figsize):
    '''Views a patch images with patches from a 1D list'''

    num_cols = self.output_dim[0]
    num_rows = self.output_dim[1]

    plt.figure(figsize=figsize)

    for x in range(self.num_patches): 
      plt.subplot(self.rows, self.rows, x+1)
      plt.imshow(input_list[x], interpolation="nearest", vmin=self.vmin, vmax=self.vmax)
    

