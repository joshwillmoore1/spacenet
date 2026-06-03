import pandas as pd
import os
import spacenet.datasets as npcd

def load_dataset(dataset_name='Spiral'):
    """
    Loads a point cloud dataset for testing and demonstration purposes. 
    The function supports multiple predefined datasets, such as 'spiral', 'cylinder', and 'sphere'. 
    Each dataset is stored as a CSV file containing the coordinates of the points in the point cloud, along with any additional attributes relevant to the dataset.
    
    Parameters
    ----------
    dataset_name : str
        The name of the dataset to load. Available datasets: ['spiral', 'cylinder', 'sphere']. Default is 'Spiral'.
    
    Returns
    -------
    pointcloud_dataframe : pandas.DataFrame
        A DataFrame containing the point cloud data for the specified dataset. The DataFrame should have columns corresponding to the coordinates of the points (e.g., 'x', 'y', 'z') and any additional attributes relevant to the dataset.
    
    Examples
    --------
    
    You can use the `load_dataset` function to load different point cloud datasets for testing and demonstration purposes. Below is an example of how to load the Spiral dataset and display the first few rows of the resulting DataFrame.
    
    .. code-block:: python
    
        import spacenet as sn

        # get data from the Spiral dataset
        sprial_df = sn.datasets.load_dataset('spiral')
        
        print(sprial_df.head())
    
    """
    
    if dataset_name.lower() == 'spiral':

        pointcloud_dataframe = pd.read_csv(os.path.dirname(npcd.__file__)+'/point_clouds/spiral_domain.csv')
        
    elif dataset_name.lower() == 'cylinder':
        
        pointcloud_dataframe = pd.read_csv(os.path.dirname(npcd.__file__)+'/point_clouds/cylinder_domain.csv')
        
    elif dataset_name.lower() == 'sphere':
        
        pointcloud_dataframe = pd.read_csv(os.path.dirname(npcd.__file__)+'/point_clouds/sphere_domain.csv')
            
        
    else:
        raise ValueError(f"Dataset '{dataset_name}' not found. Available datasets: ['spiral', 'cylinder', 'sphere']")
    
    return pointcloud_dataframe