import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio
import geopandas as gpd
from rasterio.mask import mask
import os
from shapely.geometry import mapping

# Set the style for plots
sns.set_style('whitegrid')

# Load MPI data
mpi_data = pd.read_csv('label_map.csv')

# Clean up the MPI data - convert percentage strings to floats
mpi_data['Headcount Ratio 2019/21 (V)'] = mpi_data['Headcount Ratio 2019/21 (V)'].str.rstrip('%').astype('float') / 100

# Load India shapefile with state boundaries
try:
    india_states = gpd.read_file('in.json')
    print(f"Loaded {len(india_states)} states/UTs from shapefile")
except Exception as e:
    print(f"Error loading shapefile: {e}")
    print("Proceeding with just MPI data analysis")
    india_states = None

# Try to load the nightlight TIF file
try:
    with rasterio.open('nightlight_india_2020.tif') as src:
        nightlight_data = src.read(1)
        print(f"Loaded nightlight data with shape: {nightlight_data.shape}")
        
        # Calculate average nightlight intensity per state if shapefile is available
        if india_states is not None:
            # Make sure CRS matches
            india_states = india_states.to_crs(src.crs)
            
            # Create a column to store average nightlight values
            state_nightlight_values = []
            state_names = []
            
            for idx, state in india_states.iterrows():
                try:
                    # Get state name
                    state_name = state['NAME_1'] if 'NAME_1' in state else state['name'] if 'name' in state else f"State_{idx}"
                    state_names.append(state_name)
                    
                    # Mask the raster with the state geometry
                    state_geom = [mapping(state.geometry)]
                    out_image, out_transform = mask(src, state_geom, crop=True, nodata=0)
                    
                    # Calculate average nightlight value for this state
                    avg_value = np.mean(out_image[out_image > 0]) if np.any(out_image > 0) else 0
                    state_nightlight_values.append(avg_value)
                    print(f"Processed {state_name}: Avg nightlight = {avg_value:.2f}")
                except Exception as e:
                    print(f"Error processing state {idx}: {e}")
                    state_nightlight_values.append(0)
            
            # Create a dataframe with state names and nightlight values
            nightlight_df = pd.DataFrame({
                'State/UT': state_names,
                'Average Nightlight': state_nightlight_values
            })
            
            # Merge with MPI data
            combined_data = pd.merge(mpi_data, nightlight_df, on='State/UT', how='left')
            print(f"Combined data shape: {combined_data.shape}")
            
            # Plot the relationship between nightlight intensity and poverty
            plt.figure(figsize=(12, 8))
            
            # Create scatter plot
            plt.subplot(2, 2, 1)
            sns.scatterplot(x='Average Nightlight', y='MPI 2019/21 (V)', data=combined_data, s=100, alpha=0.7)
            plt.title('Nightlight Intensity vs. MPI')
            plt.xlabel('Average Nightlight Intensity')
            plt.ylabel('Multidimensional Poverty Index (MPI)')
            
            # Add state labels to the scatter plot
            for i, row in combined_data.iterrows():
                plt.annotate(row['State/UT'], 
                            (row['Average Nightlight'], row['MPI 2019/21 (V)']),
                            xytext=(5, 5), textcoords='offset points')
            
            # Create scatter plot for headcount ratio
            plt.subplot(2, 2, 2)
            sns.scatterplot(x='Average Nightlight', y='Headcount Ratio 2019/21 (V)', 
                           data=combined_data, s=100, alpha=0.7)
            plt.title('Nightlight Intensity vs. Poverty Headcount Ratio')
            plt.xlabel('Average Nightlight Intensity')
            plt.ylabel('Poverty Headcount Ratio')
            
            # Add state labels
            for i, row in combined_data.iterrows():
                plt.annotate(row['State/UT'], 
                            (row['Average Nightlight'], row['Headcount Ratio 2019/21 (V)']),
                            xytext=(5, 5), textcoords='offset points')
            
            # Create a heatmap of states by MPI and nightlight
            plt.subplot(2, 2, 3)
            combined_data_sorted = combined_data.sort_values('MPI 2019/21 (V)', ascending=False)
            sns.barplot(x='State/UT', y='MPI 2019/21 (V)', data=combined_data_sorted.head(15))
            plt.title('Top 15 States by MPI')
            plt.xticks(rotation=90)
            
            # Create a correlation heatmap
            plt.subplot(2, 2, 4)
            correlation = combined_data[['MPI 2019/21 (V)', 'Headcount Ratio 2019/21 (V)', 'Average Nightlight']].corr()
            sns.heatmap(correlation, annot=True, cmap='coolwarm')
            plt.title('Correlation Matrix')
            
            plt.tight_layout()
            plt.savefig('poverty_nightlight_analysis.png')
            plt.show()
            
            # Save the combined data
            combined_data.to_csv('poverty_nightlight_combined.csv', index=False)
            
            # Add machine learning models to predict poverty based on nightlight data
            print("\n===== PREDICTIVE MODELING OF POVERTY USING NIGHTLIGHT DATA =====")
            
            try:
                # Import the poverty models module
                import poverty_models
                
                # Define features and target variables
                # We'll predict both MPI and Headcount Ratio
                X = combined_data[['Average Nightlight']]
                y_mpi = combined_data['MPI 2019/21 (V)']
                y_headcount = combined_data['Headcount Ratio 2019/21 (V)']
                
                # Train and evaluate models
                mpi_results, headcount_results, best_mpi_model, best_headcount_model = poverty_models.train_and_evaluate_models(
                    X, y_mpi, y_headcount
                )
                
                # Only proceed if we have valid models
                if best_mpi_model is not None and best_headcount_model is not None:
                    # Save the best models
                    poverty_models.save_models(best_mpi_model, best_headcount_model)
                    
                    # Visualize model results
                    poverty_models.visualize_model_results(
                        mpi_results, headcount_results, y_mpi, y_headcount, 
                        best_mpi_model, best_headcount_model
                    )
                    
                    # Example prediction
                    poverty_models.example_prediction()
                else:
                    print("Model training failed. Unable to proceed with predictions.")
            except Exception as e:
                print(f"Error in predictive modeling: {e}")
                print("Continuing with the rest of the analysis...")
            
            print("\nAnalysis complete. All results saved.")
        else:
            print("No state boundaries available for spatial analysis")
            
            # Just analyze MPI data without nightlight
            plt.figure(figsize=(12, 6))
            
            # Plot MPI by state
            plt.subplot(1, 2, 1)
            mpi_sorted = mpi_data.sort_values('MPI 2019/21 (V)', ascending=False)
            sns.barplot(x='State/UT', y='MPI 2019/21 (V)', data=mpi_sorted.head(15))
            plt.title('Top 15 States by MPI')
            plt.xticks(rotation=90)
            
            # Plot Headcount Ratio by state
            plt.subplot(1, 2, 2)
            headcount_sorted = mpi_data.sort_values('Headcount Ratio 2019/21 (V)', ascending=False)
            sns.barplot(x='State/UT', y='Headcount Ratio 2019/21 (V)', data=headcount_sorted.head(15))
            plt.title('Top 15 States by Poverty Headcount Ratio')
            plt.xticks(rotation=90)
            
            plt.tight_layout()
            plt.savefig('poverty_analysis.png')
            plt.show()
            
            print("Analysis complete. Results saved to 'poverty_analysis.png'")
            
except Exception as e:
    print(f"Error loading nightlight data: {e}")
    print("Proceeding with just MPI data analysis")
    
    # Just analyze MPI data without nightlight
    plt.figure(figsize=(12, 6))
    
    # Plot MPI by state
    plt.subplot(1, 2, 1)
    mpi_sorted = mpi_data.sort_values('MPI 2019/21 (V)', ascending=False)
    sns.barplot(x='State/UT', y='MPI 2019/21 (V)', data=mpi_sorted.head(15))
    plt.title('Top 15 States by MPI')
    plt.xticks(rotation=90)
    
    # Plot Headcount Ratio by state
    plt.subplot(1, 2, 2)
    headcount_sorted = mpi_data.sort_values('Headcount Ratio 2019/21 (V)', ascending=False)
    sns.barplot(x='State/UT', y='Headcount Ratio 2019/21 (V)', data=headcount_sorted.head(15))
    plt.title('Top 15 States by Poverty Headcount Ratio')
    plt.xticks(rotation=90)
    
    plt.tight_layout()
    plt.savefig('poverty_analysis.png')
    plt.show()
    
    print("Analysis complete. Results saved to 'poverty_analysis.png'")