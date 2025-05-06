from keras.saving import register_keras_serializable
import tensorflow as tf


class SingleHeadAttention(tf.keras.layers.Layer):
    def __init__(self, hidden_dim):
        """
        Single-head attention layer.

        Args:
            hidden_dim: Dimensionality for queries, keys, and values.
        """
        super(SingleHeadAttention, self).__init__()
        # hidden_dim: The dimension of the projected tensors.
        self.hidden_dim = hidden_dim
        # W_value: A dense layer mapping inputs to tensors with dimension hidden_dim
        self.W_query = tf.keras.layers.Dense(hidden_dim)
        # W_key: A dense layer mapping inputs to tensors with dimension hidden_dim
        self.W_key = tf.keras.layers.Dense(hidden_dim)
        # W_value: A dense layer mapping inputs to tensors with dimension hidden_dim
        self.W_value = tf.keras.layers.Dense(hidden_dim)

    def call(self, inputs):
        """
        Forward pass for single-head attention.

        Args:
            inputs: Tensor of shape (batch_size, seq_length, input_dim).

        Returns:
            attention_output: Attention output of shape (batch_size, seq_length, hidden_dim).
        """
        # Linear Projections
        Q = self.W_query(inputs)
        K = self.W_key(inputs)
        V = self.W_value(inputs)

        # Scaled Dot-Product Attention
        attention_scores = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(
            tf.cast(self.hidden_dim, tf.float32)
        )
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        attention_output = tf.matmul(attention_weights, V)

        return attention_output


class MultiHeadAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim, head_dim, num_heads):
        """
        Multi-head attention layer.

        Args:
            input_dim: Dimensionality of the input and final output.
            head_dim: Dimensionality for each attention head.
            num_heads: Number of attention heads.
        """
        # Each head is an instance of your single-head attention layer,
        # configured with a projection output dimensionality of head_dim.
        super(MultiHeadAttentionLayer, self).__init__()
        self.input_dim = input_dim
        self.head_dim = head_dim
        self.num_heads = num_heads

        # Create a list of attention heads. Each head is an instance of your single-head attention layer,
        # configured with a projection output dimensionality of head_dim.
        self.heads = [SingleHeadAttention(head_dim) for _ in range(num_heads)]

        # Define a Dense layer that projects the concatenated outputs
        # back to the original input dimension (input_dim).
        self.output_projection = tf.keras.layers.Dense(input_dim)

    def call(self, inputs):
        """
        Forward pass for multi-head attention.

        Args:
            inputs: Tensor of shape (batch_size, seq_length, input_dim).

        Returns:
            multihead_output: Multi-head attention output of shape (batch_size, seq_length, input_dim).
        """
        head_outputs = [head(inputs) for head in self.heads]
        concat_output = tf.concat(head_outputs, axis=-1)
        multihead_output = self.output_projection(concat_output)

        return multihead_output


class TransformerEncoderBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        embed_dim,
        attention_dim,
        feedforward_dim,
        num_heads,
        ffn_dropout_rate=0.1,
        attn_dropout_rate=0.1,
        epsilon=1e-5,
    ):
        """
        A single Transformer encoder block with multi-head attention and feed-forward network.

        Args:
            embed_dim: Dimensionality of the input embeddings.
            attention_dim: Dimensionality used in each attention head.
            feedforward_dim: Dimensionality of the intermediate feed-forward layer.
            num_heads: Number of attention heads.
        """
        super(TransformerEncoderBlock, self).__init__()
        # for call feature later
        self.attn_norm = tf.keras.layers.LayerNormalization(epsilon=epsilon)

        # Create a MultiHeadAttentionLayer that processes the inputs
        # to produce attn_output = MultiHeadAttentionLayer(inputs)
        self.attention = MultiHeadAttentionLayer(
            input_dim=embed_dim, head_dim=attention_dim, num_heads=num_heads
        )
        # for call feature later
        self.attn_dropout = tf.keras.layers.Dropout(attn_dropout_rate)

        # Build a Sequential block consisting of:
        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.LayerNormalization(
                    epsilon=epsilon
                ),  # LayerNormalization with epsilon = 1e-5
                tf.keras.layers.Dense(
                    feedforward_dim, activation=tf.nn.gelu
                ),  # A Dense layer with GELU activation mapping to feedforward_dim units
                tf.keras.layers.Dropout(
                    ffn_dropout_rate
                ),  # A Dropout layer with ffn_dropout_rate.
                tf.keras.layers.Dense(
                    embed_dim
                ),  # A Dense layer mapping back to embed_dim
                tf.keras.layers.Dropout(
                    ffn_dropout_rate
                ),  # A Dropout layer with ffn_dropout_rate.
            ]
        )

    def call(self, inputs, training=False):
        """
        Forward pass for a single Transformer encoder block.
        """
        # Compute the Multi-Head Attention
        normalized_inputs = self.attn_norm(
            inputs
        )  # Apply layer norm to inputs with epsilon = 1e-5
        attn_output = self.attention(
            normalized_inputs
        )  # Apply the multi-head attention layer
        attn_output = self.attn_dropout(
            attn_output, training=training
        )  # Apply Dropout with rate attn_dropout_rate:
        x = inputs + attn_output  # Add a residual connection:

        # Process through the Feed-Forward Network:
        ffn_output = self.ffn(x, training=training)  # Compute the feed-forward output
        output = x + ffn_output  # Apply a second residual connection

        return output


class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(
        self, embed_dim, attention_dim, feedforward_dim, num_heads, num_blocks
    ):
        """
        A stack of Transformer encoder blocks.

        Args:
            embed_dim: Dimensionality of the input embeddings.
            attention_dim: Dimensionality used in each attention head.
            feedforward_dim: Dimensionality of the intermediate feed-forward layer.
            num_heads: Number of attention heads.
            num_blocks: Number of encoder blocks.
        """

        super(TransformerEncoder, self).__init__()
        # BEGIN SOLUTION
        self.encoder_blocks = [
            TransformerEncoderBlock(
                embed_dim=embed_dim,
                attention_dim=attention_dim,
                feedforward_dim=feedforward_dim,
                num_heads=num_heads,
            )
            for _ in range(num_blocks)
        ]
        # END SOLUTION

    def call(self, inputs):
        """
        Forward pass for the Transformer encoder.

        Args:
            inputs: Tensor of shape (batch_size, seq_length, embed_dim).

        Returns:
            output: The final output of the encoder (batch_size, seq_length, embed_dim).
        """
        x = inputs

        for block in self.encoder_blocks:
            x = block(x)

        return x


class PatchEmbedding(tf.keras.layers.Layer):
    def __init__(self, image_size, patch_size, input_channels, embed_dim):
        """
        Converts an image into a sequence of patch embeddings.

        Args:
            image_size: Size (height/width) of the input image (assumed square).
            patch_size: Size of each (square) patch.
            input_channels: Number of channels in the input image.
            embed_dim: Dimensionality of the patch embeddings.
        """

        super(PatchEmbedding, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.input_channels = input_channels
        self.embed_dim = embed_dim

        assert (
            image_size % patch_size == 0
        ), f"Image size {image_size} must be divisible by patch size {patch_size}"

        self.num_patches = (image_size // patch_size) ** 2

        self.projection = tf.keras.layers.Dense(embed_dim)

    def call(self, images):
        """
        Forward pass for patch embedding.

        Args:
            images: Tensor of shape (batch_size, height, width, input_channels).

        Returns:
            A tensor of shape (batch_size, num_patches, embed_dim).
        """
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )

        patches = tf.reshape(
            patches,
            [
                tf.shape(patches)[0],
                -1,
                self.patch_size * self.patch_size * self.input_channels,
            ],
        )
        patch_embeddings = self.projection(patches)

        return patch_embeddings


# Decorator is to be able to save the model
@register_keras_serializable()
class VisionTransformer(tf.keras.Model):
    # We'll give you the init ;)
    def __init__(
        self,
        num_classes,
        patch_size,
        num_heads,
        num_blocks,
        embed_dim,
        attention_dim,
        feedforward_dim,
        input_size=(28, 28, 1),
    ):
        """
        Vision Transformer model implementation.

        Args:
            num_classes: Number of output classes.
            patch_size: Size of each patch.
            num_heads: Number of attention heads.
            num_blocks: Number of Transformer encoder blocks.
            embed_dim: Dimensionality of the patch/position embeddings.
            attention_dim: Dimensionality used in each attention head.
            feedforward_dim: Dimensionality of the intermediate feed-forward layer.
            input_size: Dimensionality of the input
        """
        super(VisionTransformer, self).__init__()
        image_size = input_size[0]
        input_channels = input_size[-1]
        self.patch_embedding = PatchEmbedding(
            image_size, patch_size, input_channels, embed_dim
        )
        self.num_patches = (image_size // patch_size) ** 2
        self.embed_dim = embed_dim

        # Positional embedding for each patch
        self.position_embedding = tf.keras.layers.Embedding(
            input_dim=self.num_patches, output_dim=embed_dim
        )

        # Learnable class token used for classification
        self.cls = self.add_weight(
            "cls", shape=(1, 1, embed_dim), initializer=tf.random_normal_initializer()
        )

        self.transformer_encoder = TransformerEncoder(
            embed_dim=embed_dim,
            attention_dim=attention_dim,
            feedforward_dim=feedforward_dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
        )

        # Classification head: LayerNorm -> Dense
        self.classification_head = tf.keras.Sequential(
            [
                tf.keras.layers.LayerNormalization(epsilon=1e-5),
                tf.keras.layers.Dense(num_classes),
            ]
        )

    def call(self, images):
        """
        Forward pass through the Vision Transformer model.

        Args:
            images: Tensor of shape (batch_size, height, width, input_channels).

        Returns:
            logits: Logits of shape (batch_size, num_classes).
        """
        batch_size = tf.shape(images)[0]
        # Compute Patch Embeddings: Pass images through self.patch_embedding to
        # obtain patch_embeddings of shape (batch_size, num_patches, embed_dim)
        patch_embeddings = self.patch_embedding(images)

        # Add Positional Embeddings:
        # Generate a sequence of positions (of length num_patches) and map them
        # through self.position_embedding to obtain positional embeddings:
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        positional_embeddings = self.position_embedding(positions)
        # positional_embeddings of shape (1, num_patches, embed_dim)
        positional_embeddings = tf.expand_dims(positional_embeddings, axis=0)
        # Add these positional embeddings to patch_embeddings elementwise.
        transformer_input = patch_embeddings + positional_embeddings

        # Prepend the Class Token:
        # Broadcast the learnable class token to match the batch size,
        # resulting in a tensor of shape (batch_size, 1, embed_dim).
        cls_token = tf.broadcast_to(self.cls, [batch_size, 1, self.embed_dim])
        # Concatenate this class token with the positional-enhanced
        # patch embeddings along the sequence dimension, yielding:
        transformer_input = tf.concat([cls_token, transformer_input], axis=1)

        # Apply the Transformer Encoder:
        # Pass transformer_input through self.transformer_encoder which outputs encoder_output
        # with shape (batch_size, num_patches + 1, embed_dim)
        encoded = self.transformer_encoder(transformer_input)

        # Classification:
        # The classification head uses the embedding corresponding to the class token (index 0)
        # to compute logits, with shape(batch_size, num_classes)
        logits = self.classification_head(encoded[:, 0, :])

        # The model thus returns the logits that represent the class scores for each input image.
        return logits
