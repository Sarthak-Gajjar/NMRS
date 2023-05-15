import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import itertools
import threading

class MovieRecommender:
    def __init__(self, data_filename):
        """
        Initialize MovieRecommender with data from a specified CSV file
        and create the GUI window.
        """
        self.data = self.load_data(data_filename)
    
        # Set up the GUI window
        self.window = tk.Tk()
        self.window.title('Netflix Recommender')
        self.window.configure(background='lightblue')  # Set the background color
    
        # Add a label for the input box
        input_label = tk.Label(self.window, text="Enter movie title:", bg='lightblue', fg='black', font=('Arial', 14, 'bold'))
        input_label.pack(pady=10)  # Added padding for better layout
    
        # Create the input box
        self.input_box = tk.Entry(self.window, width=50)
        self.input_box.pack(pady=10)  # Added padding for better layout
    
        # Add a label for the output box
        output_label = tk.Label(self.window, text="Top 10 recommendations:", bg='lightblue', fg='black', font=('Arial', 14, 'bold'))
        output_label.pack(pady=10)  # Added padding for better layout
    
        # Create the button with recommend_movies as its command
        recommend_button = tk.Button(self.window, text='Recommend', command=self.recommend, bg='black', fg='white', font=('Arial', 12, 'bold'))
        recommend_button.pack(pady=10)  # Added padding for better layout
    
        self.output_box = tk.Text(self.window, width=60, height=30)
        self.output_box.pack(pady=10)  # Added padding for better layout


    def load_data(self, filename):
        """
        Load and return data from a CSV file.
        """
        return pd.read_csv(filename)

    def create_movie_features(self, data):
        """
        Create binary features for each movie in the dataset.
        """
        movie_features = pd.DataFrame()

        # One-hot encode features for each movie
        for feature in ['Cast', 'Production Country', 'Genres']:
            one_hot_encoded_features = pd.get_dummies(data[feature], prefix=feature)
            movie_features = pd.concat([movie_features, one_hot_encoded_features], axis=1)

        return movie_features

    def calculate_cosine_similarity(self, movie_features1, movie_features2):
        """
        Calculate and return the cosine similarity between two sets of movie features.
        """
        dot_product = np.dot(movie_features1, movie_features2)
        norm1 = np.linalg.norm(movie_features1)
        norm2 = np.linalg.norm(movie_features2)
        return dot_product / (norm1 * norm2)

    def recommend_movies(self, search):
        """
        Check if the search term is in the dataset and recommend movies based on it.
        """
        # Verify if the search term is in the dataset
        if search not in self.data['Title'].values:
            return None, 'Title not in dataset. Please check spelling.'

        # Get the index of the search term in the dataset
        idx = self.data[self.data['Title'] == search].index.item()

        # Create binary features for all movies
        movie_features = self.create_movie_features(self.data)

        # Calculate the cosine similarity between the search term and all other movies
        cos_sims = []
        for i in range(len(self.data)):
            cos_sim = self.calculate_cosine_similarity(movie_features.iloc[idx], movie_features.iloc[i])
            cos_sims.append(cos_sim)

        # Sort the movies by cosine similarity, excluding the search term, and return the top 5
        results = self.data.copy()
        results['cos_sim'] = cos_sims
        results = results.sort_values('cos_sim', ascending=False)
        results = results[results['Title'] != search]
        return idx, results.head(10)

    def animate_gif(self, gif, label):
        """
        Animate the given GIF by updating the label with the next frame
        in the sequence.
        """
        frame = next(gif)  # Get the next frame from the iterator
        label.configure(image=frame)
        label.image = frame
        self.window.after(50, self.animate_gif, gif, label)  # Schedule the next frame update

    def load_gif(self, path):
        """
        Load the frames of a GIF and return an iterator.
        """
        gif_frames = []
        gif = Image.open(path)

        try:
            while True:
                frame = ImageTk.PhotoImage(gif.copy())
                gif_frames.append(frame)
                gif.seek(len(gif_frames))  # Move to the next frame in the GIF
        except EOFError:
            pass  # Reached the end of the GIF

        return itertools.cycle(gif_frames)  # Return an iterator that cycles through the frames

    def recommend(self):
        """
        Get the user's input and recommend movies based on it.
        """
        # Clear the output box
        self.output_box.delete(1.0, tk.END)
    
        # Show the animated loading GIF
        loading_gif = self.load_gif('monkey_working.gif')  # Specify your own loading gif file here
        loading_label = tk.Label(self.window)
        loading_label.pack()
        self.animate_gif(loading_gif, loading_label)
    
        # Run the recommendation process in a separate thread
        threading.Thread(target=self.get_recommendations, args=(loading_label,)).start()
    
    def get_recommendations(self, loading_label):
        search = self.input_box.get()
        original_index, recommendations = self.recommend_movies(search)
    
        # Remove the loading GIF
        loading_label.pack_forget()
    
        if recommendations is None:
            messagebox.showerror("Error", "Title not in dataset. Please check spelling.")
            return
    
        # Display the original movie title and year
        if original_index is not None:
            original_title = self.data.loc[original_index, 'Title']
            original_year = int(self.data.loc[original_index, 'Release Date'])  # convert to integer
            self.output_box.insert(tk.END, f"Original movie: \n \n{original_title} ({original_year})\n\n")
        
            # Print the recommendations in the output box    
            self.output_box.insert(tk.END, f"Recommended movies: \n \n")
            for index, movie in recommendations.iterrows():
                movie_title = movie['Title']
                movie_year = int(movie['Release Date'])  # convert to integer
                self.output_box.insert(tk.END, f"{movie_title} ({movie_year})\n")
    
                
        else:
            messagebox.showerror("Error", "Title not in dataset. Please check spelling.")

    def run(self):
        """
        Start the main loop for the GUI.
        """
        self.window.mainloop()


if __name__ == '__main__':
    recommender = MovieRecommender('netflixData.csv')
    recommender.run()