import pandas as pd
import os
from datetime import datetime


class DataManager:
    """
    Data Manager for FeedIQ Platform
    
    Handles loading, saving, and managing feedback data
    with proper error handling and data validation.
    """
    
    def __init__(self, path='dataset.csv'):
        """
        Initialize Data Manager
        
        Args:
            path (str): Path to the CSV file for storing feedback
        """
        self.path = path
        self._ensure_file_exists()
    
    def _ensure_file_exists(self):
        """
        Ensure the CSV file exists, create if it doesn't
        """
        if not os.path.exists(self.path):
            # Create initial dataset with headers
            df = pd.DataFrame(columns=['id', 'text', 'timestamp'])
            df.to_csv(self.path, index=False)
            print(f"✅ Created new dataset file: {self.path}")
    
    def load(self):
        """
        Load feedback data from CSV file
        
        Returns:
            DataFrame: Feedback data with id, text, and timestamp columns
        """
        try:
            df = pd.read_csv(self.path)
            
            # Ensure required columns exist
            if 'id' not in df.columns or 'text' not in df.columns:
                raise ValueError("CSV must contain 'id' and 'text' columns")
            
            # Add timestamp column if it doesn't exist
            if 'timestamp' not in df.columns:
                df['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                df.to_csv(self.path, index=False)
            
            return df
        
        except FileNotFoundError:
            print(f"⚠️ File not found: {self.path}. Creating new dataset.")
            self._ensure_file_exists()
            return self.load()
        
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            # Return empty dataframe with correct structure
            return pd.DataFrame(columns=['id', 'text', 'timestamp'])
    
    def add_feedback(self, text, save=True):
        """
        Add new feedback to the dataset
        
        Args:
            text (str): Feedback text to add
            save (bool): Whether to save to CSV immediately
            
        Returns:
            bool: Success status
        """
        try:
            # Validate input
            if not text or not text.strip():
                print("⚠️ Cannot add empty feedback")
                return False
            
            df = pd.read_csv(self.path)
            
            # Ensure timestamp column exists
            if 'timestamp' not in df.columns:
                df['timestamp'] = ''
            
            # Generate new ID
            new_id = len(df) + 1
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Add new row using concat (safer than loc)
            new_row = pd.DataFrame({
                'id': [new_id],
                'text': [text.strip()],
                'timestamp': [timestamp]
            })
            
            df = pd.concat([df, new_row], ignore_index=True)
            
            if save:
                df.to_csv(self.path, index=False)
                print(f"✅ Feedback #{new_id} added successfully")
            
            return True
        
        except Exception as e:
            print(f"❌ Error adding feedback: {e}")
            return False
    
    def delete_feedback(self, feedback_id):
        """
        Delete feedback by ID
        
        Args:
            feedback_id (int): ID of feedback to delete
            
        Returns:
            bool: Success status
        """
        try:
            df = pd.read_csv(self.path)
            
            if feedback_id not in df['id'].values:
                print(f"⚠️ Feedback ID {feedback_id} not found")
                return False
            
            df = df[df['id'] != feedback_id]
            df.to_csv(self.path, index=False)
            print(f"✅ Feedback #{feedback_id} deleted")
            return True
        
        except Exception as e:
            print(f"❌ Error deleting feedback: {e}")
            return False
    
    def update_feedback(self, feedback_id, new_text):
        """
        Update existing feedback text
        
        Args:
            feedback_id (int): ID of feedback to update
            new_text (str): New feedback text
            
        Returns:
            bool: Success status
        """
        try:
            df = pd.read_csv(self.path)
            
            if feedback_id not in df['id'].values:
                print(f"⚠️ Feedback ID {feedback_id} not found")
                return False
            
            df.loc[df['id'] == feedback_id, 'text'] = new_text.strip()
            df.to_csv(self.path, index=False)
            print(f"✅ Feedback #{feedback_id} updated")
            return True
        
        except Exception as e:
            print(f"❌ Error updating feedback: {e}")
            return False
    
    def get_stats(self):
        """
        Get basic statistics about the dataset
        
        Returns:
            dict: Statistics including count, average length, etc.
        """
        try:
            df = pd.read_csv(self.path)
            
            stats = {
                'total_feedbacks': len(df),
                'avg_text_length': df['text'].str.len().mean() if len(df) > 0 else 0,
                'latest_feedback': df['timestamp'].max() if 'timestamp' in df.columns and len(df) > 0 else None
            }
            
            return stats
        
        except Exception as e:
            print(f"❌ Error getting stats: {e}")
            return {
                'total_feedbacks': 0,
                'avg_text_length': 0,
                'latest_feedback': None
            }
    
    def export_to_csv(self, output_path):
        """
        Export data to a different CSV file
        
        Args:
            output_path (str): Path for the export file
            
        Returns:
            bool: Success status
        """
        try:
            df = pd.read_csv(self.path)
            df.to_csv(output_path, index=False)
            print(f"✅ Data exported to {output_path}")
            return True
        
        except Exception as e:
            print(f"❌ Error exporting data: {e}")
            return False
    
    def bulk_add_feedback(self, feedback_list):
        """
        Add multiple feedbacks at once
        
        Args:
            feedback_list (list): List of feedback text strings
            
        Returns:
            int: Number of feedbacks successfully added
        """
        success_count = 0
        
        try:
            df = pd.read_csv(self.path)
            
            # Ensure timestamp column exists
            if 'timestamp' not in df.columns:
                df['timestamp'] = ''
            
            for text in feedback_list:
                if text and text.strip():
                    new_id = len(df) + 1
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    
                    new_row = pd.DataFrame({
                        'id': [new_id],
                        'text': [text.strip()],
                        'timestamp': [timestamp]
                    })
                    
                    df = pd.concat([df, new_row], ignore_index=True)
                    success_count += 1
            
            df.to_csv(self.path, index=False)
            print(f"✅ Added {success_count} feedbacks in bulk")
            return success_count
        
        except Exception as e:
            print(f"❌ Error in bulk add: {e}")
            return success_count