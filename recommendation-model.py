import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

class VAERecommender:
    """
    Variational Autoencoder (VAE) based recommendation system for Kuku FM
    
    This model combines collaborative filtering with content-based approaches in a 
    generative framework, allowing for both accurate recommendations and content exploration.
    """
    
    def __init__(self, 
                 user_dim=128,         # Dimension of user embedding
                 content_dim=256,      # Dimension of content embedding  
                 latent_dim=64,        # Dimension of latent space
                 hidden_dims=[512, 256],  # Hidden dimensions of encoder/decoder
                 dropout_rate=0.2,
                 learning_rate=0.001):
        
        self.user_dim = user_dim
        self.content_dim = content_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        # Build the VAE components
        self.encoder = None
        self.decoder = None
        self.vae = None
        
        # Additional components
        self.user_embedding = None
        self.content_embedding = None
        self.scaler = StandardScaler()
        self.content_encoder = None
        
    def preprocess_data(self, user_data, content_data):
        """
        Preprocess user interaction and content data
        
        Args:
            user_data: DataFrame containing user listening history 
                      (user_id, content_id, listen_duration, completion_percentage, etc.)
            content_data: DataFrame containing content features
                      (content_id, title, creator, category, duration, transcript, etc.)
        """
        # Process user interaction data
        user_matrix = self._create_user_interaction_matrix(user_data)
        
        # Process content data
        content_features = self._extract_content_features(content_data)
        
        return user_matrix, content_features
    
    def _create_user_interaction_matrix(self, user_data):
        """Create user-content interaction matrix with various engagement signals"""
        # Create interaction matrix with multi-signal approach
        # This captures not just if a user listened but how they engaged
        
        # Example signals:
        # - Completion rate (percentage of content listened to)
        # - Listen count (how many times content was played)
        # - Like/bookmark actions
        # - Time spent (total duration listened)
        # - Recency (when was the last listen)
        
        # Pivot to create user-content matrix
        completion_matrix = user_data.pivot(
            index='user_id', 
            columns='content_id',
            values='completion_percentage'
        ).fillna(0)
        
        listen_count_matrix = user_data.pivot(
            index='user_id', 
            columns='content_id',
            values='listen_count'
        ).fillna(0)
        
        # Normalize listen counts
        listen_count_matrix = listen_count_matrix / listen_count_matrix.max().max()
        
        # Combine signals (weighted approach)
        interaction_matrix = (0.7 * completion_matrix) + (0.3 * listen_count_matrix)
        
        # More advanced: could include time decay for older listens
        
        return interaction_matrix.values
    
    def _extract_content_features(self, content_data):
        """Extract and process content features"""
        # Categorical features processing
        categorical_features = ['category', 'language', 'creator']
        self.content_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        
        categorical_encoded = self.content_encoder.fit_transform(
            content_data[categorical_features]
        )
        
        # Numerical features processing
        numerical_features = ['duration', 'popularity_score', 'avg_completion_rate']
        numerical_values = content_data[numerical_features].values
        numerical_scaled = self.scaler.fit_transform(numerical_values)
        
        # Text features processing (simplified here, would use embeddings in production)
        # In a real implementation, this would use pre-trained embeddings or transformer models
        # to generate embeddings from content transcripts or descriptions
        
        # For demonstration purposes, representing as random vectors
        # In production: text_embeddings = text_encoder_model(content_data['transcript'])
        text_embedding_dim = 128
        text_embeddings = np.random.normal(size=(len(content_data), text_embedding_dim))
        
        # Combine all features
        combined_features = np.hstack([
            categorical_encoded,
            numerical_scaled,
            text_embeddings
        ])
        
        return combined_features
    
    def build_model(self, num_users, num_content_items):
        """Build the VAE recommendation model"""
        # User embedding layer
        user_input = keras.Input(shape=(1,), name="user_input")
        self.user_embedding = layers.Embedding(
            num_users, 
            self.user_dim,
            embeddings_initializer="glorot_normal",
            name="user_embedding"
        )
        user_embedded = self.user_embedding(user_input)
        user_embedded = layers.Flatten()(user_embedded)
        
        # Content embedding
        content_input = keras.Input(shape=(self.content_features.shape[1],), name="content_input")
        content_dense = layers.Dense(self.content_dim, activation="relu")(content_input)
        content_dropout = layers.Dropout(self.dropout_rate)(content_dense)
        
        # VAE Encoder
        encoder_inputs = layers.Concatenate()([user_embedded, content_dropout])
        
        # Build encoder layers
        x = encoder_inputs
        for dim in self.hidden_dims:
            x = layers.Dense(dim, activation="relu")(x)
            x = layers.Dropout(self.dropout_rate)(x)
        
        # VAE latent space
        z_mean = layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(x)
        
        # Sampling function
        def sampling(args):
            z_mean, z_log_var = args
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon
        
        z = layers.Lambda(sampling, output_shape=(self.latent_dim,), name="z")([z_mean, z_log_var])
        
        # Build encoder model
        self.encoder = keras.Model([user_input, content_input], [z_mean, z_log_var, z], name="encoder")
        
        # VAE Decoder
        latent_inputs = keras.Input(shape=(self.latent_dim,), name="decoder_input")
        
        # Build decoder layers (reverse of encoder)
        x = latent_inputs
        for dim in reversed(self.hidden_dims):
            x = layers.Dense(dim, activation="relu")(x)
            x = layers.Dropout(self.dropout_rate)(x)
        
        # Output layer - predicting user-content interaction scores
        decoder_outputs = layers.Dense(num_content_items, activation="sigmoid")(x)
        
        # Build decoder model
        self.decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
        
        # Build end-to-end VAE model
        outputs = self.decoder(self.encoder([user_input, content_input])[2])
        self.vae = keras.Model([user_input, content_input], outputs, name="vae")
        
        # Add VAE loss
        reconstruction_loss = keras.losses.binary_crossentropy(
            keras.Input(shape=(num_content_items,)), 
            outputs
        )
        reconstruction_loss *= num_content_items
        
        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = tf.reduce_sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        
        vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
        self.vae.add_loss(vae_loss)
        
        # Compile model
        self.vae.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate))
        
        return self.vae
    
    def train(self, user_data, content_data, epochs=100, batch_size=128, validation_split=0.2):
        """Train the model on user interaction and content data"""
        # Preprocess data
        user_matrix, self.content_features = self.preprocess_data(user_data, content_data)
        
        # Get dimensions
        num_users = len(np.unique(user_data['user_id']))
        num_content_items = len(np.unique(user_data['content_id']))
        
        # Build model
        self.build_model(num_users, num_content_items)
        
        # Prepare training data
        user_ids = np.array(user_data['user_id'].unique())
        
        # Create user indices and content feature inputs
        X_users = []
        X_content = []
        y_interaction = []
        
        for i, user_id in enumerate(user_ids):
            user_interactions = user_matrix[i]
            X_users.append(np.full(num_content_items, i))
            X_content.append(self.content_features)
            y_interaction.append(user_interactions)
        
        X_users = np.concatenate(X_users)
        X_content = np.vstack(X_content)
        y_interaction = np.concatenate(y_interaction)
        
        # Train-test split
        indices = np.arange(len(X_users))
        train_indices, val_indices = train_test_split(indices, test_size=validation_split)
        
        # Train the model
        self.vae.fit(
            [X_users[train_indices], X_content[train_indices]],
            y_interaction[train_indices],
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(
                [X_users[val_indices], X_content[val_indices]],
                y_interaction[val_indices]
            ),
            callbacks=[
                keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
            ]
        )
        
        return self.vae
    
    def generate_recommendations(self, user_id, n=10, diversity_factor=0.2):
        """
        Generate personalized recommendations for a user
        
        Args:
            user_id: ID of the user
            n: Number of recommendations to generate
            diversity_factor: Controls exploration vs. exploitation (0-1)
                             Higher values increase diversity but may reduce relevance
        
        Returns:
            List of recommended content IDs
        """
        # Get user index
        user_idx = np.where(self.user_ids == user_id)[0][0]
        
        # Get user representation in latent space
        user_input = np.array([user_idx])
        content_input = self.content_features
        
        # Get latent representation
        z_mean, z_log_var, _ = self.encoder.predict([
            np.full(len(content_input), user_idx),
            content_input
        ])
        
        # Add some exploration noise based on diversity factor
        z_mean_user = z_mean.mean(axis=0)
        z_log_var_user = z_log_var.mean(axis=0)
        
        # Sample multiple points from the user's latent distribution
        num_samples = 5
        recommendations_sets = []
        
        for _ in range(num_samples):
            # Sample from latent space with controlled noise
            epsilon = np.random.normal(size=self.latent_dim) * diversity_factor
            z_sample = z_mean_user + np.exp(0.5 * z_log_var_user) * epsilon
            
            # Generate recommendations from this sample
            predicted_scores = self.decoder.predict(np.array([z_sample]))
            
            # Get top recommendations
            top_indices = np.argsort(-predicted_scores[0])[:n]
            recommendations_sets.append(top_indices)
        
        # Aggregate recommendations using a weighted approach
        # (items that appear in multiple samples get higher priority)
        recommendation_counts = {}
        for rec_set in recommendations_sets:
            for i, idx in enumerate(rec_set):
                # Weight by position (earlier = higher weight)
                weight = n - i
                if idx in recommendation_counts:
                    recommendation_counts[idx] += weight
                else:
                    recommendation_counts[idx] = weight
        
        # Sort by frequency and weight
        sorted_recommendations = sorted(
            recommendation_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Extract just the content indices
        return [idx for idx, _ in sorted_recommendations[:n]]
    
    def generate_exit_recommendations(self, user_id, time_available=15, context=None):
        """
        Generate specialized recommendations for the exit modal
        
        Args:
            user_id: ID of the user
            time_available: Estimated time in minutes the user might have
            context: Additional context dict (time of day, day of week, etc.)
            
        Returns:
            Dictionary of recommendation sets
        """
        # Default context
        if context is None:
            context = {
                'time_of_day': 'afternoon',
                'day_of_week': 'weekday',
                'last_category': 'fiction'
            }
        
        # Get base recommendations
        base_recs = self.generate_recommendations(user_id, n=20, diversity_factor=0.2)
        
        # Filter by estimated listening time
        short_content = [idx for idx in base_recs 
                        if self.content_data.loc[idx, 'duration'] <= time_available]
        
        # Get content the user was in the middle of
        in_progress = self.get_in_progress_content(user_id)
        
        # Get trending content
        trending = self.get_trending_content(time_available)
        
        # Generate final recommendation sets
        recommendations = {
            'for_you': short_content[:3],
            'continue': in_progress[:2],
            'trending': trending[:3]
        }
        
        return recommendations
    
    def get_in_progress_content(self, user_id):
        """Get content the user has started but not completed"""
        # This would connect to actual user data
        # Simplified implementation for demonstration
        return [101, 205]  # Example content IDs
    
    def get_trending_content(self, max_duration=None):
        """Get trending content, optionally filtered by duration"""
        # This would analyze recent popularity trends
        # Simplified implementation for demonstration
        return [301, 402, 103]  # Example content IDs


# Example usage with synthetic data
if __name__ == "__main__":
    # Create synthetic data
    np.random.seed(42)
    
    # User data (simplified)
    n_users = 1000
    n_content = 500
    n_interactions = 50000
    
    user_ids = np.random.randint(1, n_users + 1, n_interactions)
    content_ids = np.random.randint(1, n_content + 1, n_interactions)
    completion_pct = np.random.uniform(0, 1, n_interactions)
    listen_count = np.random.randint(1, 5, n_interactions)
    
    user_data = pd.DataFrame({
        'user_id': user_ids,
        'content_id': content_ids,
        'completion_percentage': completion_pct,
        'listen_count': listen_count
    })
    
    # Content data (simplified)
    content_ids = np.arange(1, n_content + 1)
    categories = np.random.choice(
        ['Fiction', 'Self-help', 'Business', 'Health', 'History'], 
        n_content
    )
    languages = np.random.choice(['English', 'Hindi', 'Tamil', 'Bengali'], n_content)
    creators = np.random.choice(['Creator A', 'Creator B', 'Creator C', 'Creator D'], n_content)
    durations = np.random.uniform(5, 60, n_content)  # in minutes
    popularity = np.random.uniform(0, 1, n_content)
    avg_completion = np.random.uniform(0.3, 0.9, n_content)
    
    content_data = pd.DataFrame({
        'content_id': content_ids,
        'category': categories,
        'language': languages,
        'creator': creators,
        'duration': durations,
        'popularity_score': popularity,
        'avg_completion_rate': avg_completion
    })
    
    # Initialize and train the model
    recommender = VAERecommender()
    recommender.train(user_data, content_data, epochs=5)  # Reduced epochs for demonstration
    
    # Generate recommendations for a user
    user_recommendations = recommender.generate_exit_recommendations(42)
    print("Recommendations:", user_recommendations)
