import streamlit as st
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
from sklearn.metrics import confusion_matrix
import base64

#PGD Attack
def pgd_attac(model, x, y, epsilon=0.5, alpha=0.7, num_iter=10, targeted=False, num_random_init=0, batch_size=280):
    perturbed_x = tf.identity(x)  # create a copy of the input

    for _ in range(num_iter):
        with tf.GradientTape() as tape:
            tape.watch(perturbed_x) # keep track of purturbed_x
            loss = model(perturbed_x) #calculate loss

        gradients = tape.gradient(loss, perturbed_x) # calculate gradient of loss relevent to pur_x

        if targeted:
            gradients = -gradients #reversing the direction and fine tuning pertubation

        perturbed_x = tf.clip_by_value(perturbed_x + alpha * tf.sign(gradients), x - epsilon, x + epsilon) #update purtubate x and clip to stay in a specific range
        perturbed_x = tf.clip_by_value(perturbed_x, 0, 0.5)  # ensure pixel values are in [0, 1] range

    perturbed_x = tf.stop_gradient(perturbed_x) #stop gradientflow
    return perturbed_x, y



#Boundary Attack
def identify_binary_features(data, threshold=0.05):
    num_of_samples, num_of_features = data.shape
    binary_mask = np.zeros(num_of_features, dtype=bool)

    for feature_idx in range(num_of_features):
        unique_values = np.unique(data[:, feature_idx])
        unique_ratio = len(unique_values) / num_of_samples

        if unique_ratio <= threshold:
            binary_mask[feature_idx] = True

    return binary_mask

# Define compute_norm and sel_direction functions
def compute_norm(x, x_adv):
    return np.linalg.norm(x_adv - x)

def sel_direction(x, x_adv, x_adv_p):
    norm1 = compute_norm(x, x_adv)
    norm2 = compute_norm(x, x_adv_p)
    if norm2 > norm1:
        direction = -1
    elif norm2 < norm1:
        direction = 1
    else:
        direction = 0
    return direction

def boundary_attack_tabular(model, x, y, max_iterations=50, step_size=0.1, epsilon=0.01):
    binary_mask = identify_binary_features(x)

    # Initialize the adversarial example with a perturbed version of the original input
    binary_perturbations = np.random.normal(0, step_size, size=x.shape)
    binary_perturbations *= binary_mask  # Apply the binary mask to select binary features

    continuous_perturbations = np.random.normal(0, step_size, size=x.shape)
    continuous_perturbations *= (1 - binary_mask)  # Apply the inverse of the binary mask

    total_perturbations = binary_perturbations + continuous_perturbations

    x_adv = x + total_perturbations  # Initialize x_adv with the perturbed version of x

    for _ in range(max_iterations):

        distance = compute_norm(x, x_adv)

        p_normalized = total_perturbations / np.linalg.norm(total_perturbations)

        magnitude = distance * p_normalized

        direction = sel_direction(x, x_adv, x_adv + epsilon * (x - x_adv) + magnitude) #The direction of adjustment (closer, away, or stay).

        x_adv = x_adv + direction * (epsilon * magnitude)  # Add the projected perturbation to the update rule

        # Clip feature values to appropriate ranges
        x_adv = np.clip(x_adv, 0, 1)  # Binary features
        x_adv = np.clip(x_adv, x.min(axis=0), x.max(axis=0))  # Continuous/integer features

        # Check if the adversarial example has caused a misclassification
        adv_preds = model.predict(x_adv)

        if np.argmax(adv_preds) != np.argmax(y):
            return x_adv  #Updated adversarial example after applying the adjustment.

    # If misclassifictation doesnt occur, return last recorded x_adv
    return x_adv



#Carlini Attack
def carlini_attack_binary(model, X, y, batch_size=100, epsilon=0.1, max_iterations=50, learning_rate=0.01):#creat the binary funtion
    # Process in batches
    num_batches = int(np.ceil(len(X) / batch_size))
    perturbed_Xs = []

    for i in range(num_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(X))
        X_batch = tf.identity(X[start:end])
        y_batch = tf.convert_to_tensor(y[start:end], dtype=tf.float32)
        y_batch = tf.reshape(y_batch, (-1, 1))  # Ensure y_batch shape matches prediction shape

        for _ in range(max_iterations):
            with tf.GradientTape() as tape:
                tape.watch(X_batch)
                prediction = model(X_batch)
                loss = tf.keras.losses.binary_crossentropy(y_batch, prediction)

            gradients = tape.gradient(loss, X_batch)
            X_batch -= learning_rate * gradients
            X_batch = tf.clip_by_value(X_batch, X[start:end] - epsilon, X[start:end] + epsilon)
            X_batch = tf.clip_by_value(X_batch, 0, 1)

        perturbed_Xs.append(X_batch)

    # Concatenate all batch results
    perturbed_X = tf.concat(perturbed_Xs, axis=0)
    return perturbed_X

def carlini_wagner_attack(model, x, y, c=1, lr=0.01, iterations=100):#targating a model to miss #c- control the  adv ex and comfident the ad clasifiaction
    x_adv = tf.Variable(x, dtype=tf.float32)  # Make sure x is in the correct format #ini tens variable with input data
    target = tf.constant(y, dtype=tf.float32)  # Make sure y is in the correct format#compute the loss fun in tns comp
    binary_crossentropy = tf.keras.losses.BinaryCrossentropy()

    optimizer = tf.optimizers.Adam(learning_rate=lr)#optimi the loop minimize the adveria loss
    for i in range(iterations):#update the adv ex
        with tf.GradientTape() as tape:#automatic diff
            tape.watch(x_adv)
            prediction = model(x_adv)#current ex
            loss = c * binary_crossentropy(target, prediction) + tf.norm(x_adv - x)#scal binry & l2

        gradients = tape.gradient(loss, x_adv)
        optimizer.apply_gradients([(gradients, x_adv)])
        x_adv.assign(tf.clip_by_value(x_adv, 0, 1))

    return x_adv.numpy()
    #iteratively adjeust & find model miss and changes are su or not



# Brendel and Bethge Attack
def brendel_bethge_attack(model, x_test, y_test, epsilon=0.1, iterations=100, alpha=0.02):
    x_adv = x_test.copy()  # Start with copies of the original inputs

    for i in range(iterations):
        # Introduce a random perturbation
        perturbation = np.random.normal(loc=0.0, scale=epsilon, size=x_test.shape)
        x_temp = x_adv + perturbation  # Temporarily add noise

        preds = model.predict(x_temp)  # Make predictions on the modified inputs
        preds_class = (preds > 0.5).astype(int)  # Assuming binary classification with a sigmoid output
        mask = preds_class.flatten() != y_test  # Identify where the attack changed the prediction

        # Only keep changes that successfully fooled the model
        x_adv[mask] = x_temp[mask]

        # Gradually reduce epsilon to fine-tune the adversarial examples
        epsilon *= (1 - alpha)

    return x_adv


#Frontend

# Function to encode local image to base64
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Function to set the background
def set_background(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
        background-position: center;
        opacity: 0.9; /* Adjust opacity to make the background faded */
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)


def main():
     # Set background image
    set_background('White.avif')  # Replace with your image path

    left_co, cent_co,last_co = st.columns([2, 3, 2])
    with cent_co:
        st.image('Secunet (2).png')

    left_c, cent_c, last_c = st.columns([1, 4, 1])
    with cent_c:
        st.markdown(
            """
            <h3 style='text-align: center; line-height: 1.5;'>
                Machine Learning Model<br>Vulnerability Checker
            </h3>
            """,
            unsafe_allow_html=True,
        )

    attack_option = st.sidebar.radio("Select Attack", ["PGD Attack", "Tabular Boundary Attack","Carlini Attack","Brendel & Bethge Attack"])
    #st.sidebar.image('Heart.jpg', use_column_width=True)

#If PGD Attack

    if attack_option == "PGD Attack":
        st.title("PGD Attack")
        dataset_file = st.file_uploader("Upload Dataset", type=["csv"])
        if dataset_file:
            data = pd.read_csv(dataset_file)

            # Let the user select the target column
            target_column = st.selectbox("Select Target Column", data.columns)
            X = data.drop(columns=[target_column])
            y = data[target_column]
            
            # Performing the train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_train.shape, X_test.shape, y_train.shape, y_test.shape

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            smote=SMOTE(sampling_strategy='auto', random_state=23)
            X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

            under_sampler = RandomUnderSampler(sampling_strategy='auto', random_state=23)
            X_train_combined, y_train_combined = under_sampler.fit_resample(X_train_smote, y_train_smote)

            st.write(y_train_combined.value_counts())

        model_file = st.file_uploader("Upload .h5 Model", type=["h5"])

        if model_file:
            temp_model_location = "temp_model.h5"
            with open(temp_model_location, 'wb') as out:
                out.write(model_file.read())

            loaded_model = tf.keras.models.load_model(temp_model_location)
            st.session_state.loaded_model = loaded_model
            st.session_state.X_test_scaled = X_test_scaled
            st.session_state.y_test = y_test

            original_model_accuracy = loaded_model.evaluate(X_test_scaled, y_test)[1]
            st.session_state.original_model_accuracy = original_model_accuracy

            st.write(f"Original Model Accuracy: {original_model_accuracy}")

        if st.button("Apply PGD Attack"):
            if hasattr(st.session_state, 'loaded_model'):
                loaded_model = st.session_state.loaded_model
                X_test_scaled = st.session_state.X_test_scaled
                y_test = st.session_state.y_test

                #Apply PGD Attack
            X_test_pgd, y_test_pgd = pgd_attac(loaded_model, X_test_scaled, y_test)

            # Evaluate the perturbed model
            perturbed_model_accuracy = loaded_model.evaluate(X_test_pgd, y_test_pgd)[1]
            #st.write(f"Perturbed Model Accuracy: {perturbed_model_accuracy}")

            # Display accuracy using a gauge visualization
        
            st.write(f"Original Model Accuracy:{round(original_model_accuracy, 2)}")
            st.progress(original_model_accuracy)

            st.write(f"Perturbed Model Accuracy: {round(perturbed_model_accuracy, 2)}")
            st.progress(perturbed_model_accuracy)

            st.title("Suggested Defenses")

            with st.container():
                st.subheader("Stochastic Distillation")
                st.write("It involves training a model on a mixture of clean and adversarial examples, using a stochastic process to generate perturbations. This approach aims to reduce the model's sensitivity to small input changes, ultimately bolstering its resilience against adversarial attacks like PGD.")


#If Tabular Boundary attack
    if attack_option == "Tabular Boundary Attack":
        st.title("Tabular Boundary Attack")
        dataset_file = st.file_uploader("Upload Dataset", type=["csv"])

        if dataset_file:
            data = pd.read_csv(dataset_file)

            # Let the user select the target column
            target_column = st.selectbox("Select Target Column", data.columns)
            X = data.drop(columns=[target_column])
            y = data[target_column]

            # Performing the train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_train.shape, X_test.shape, y_train.shape, y_test.shape

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            smote=SMOTE(sampling_strategy='auto', random_state=23)
            X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

            under_sampler = RandomUnderSampler(sampling_strategy='auto', random_state=23)
            X_train_combined, y_train_combined = under_sampler.fit_resample(X_train_smote, y_train_smote)

            st.write(y_train_combined.value_counts())

        model_file = st.file_uploader("Upload .h5 Model", type=["h5"])

        if model_file:
            temp_model_location = "temp_model.h5"
            with open(temp_model_location, 'wb') as out:
                out.write(model_file.read())

            loaded_model = tf.keras.models.load_model(temp_model_location)
            st.session_state.loaded_model = loaded_model
            st.session_state.X_test_scaled = X_test_scaled
            st.session_state.y_test = y_test

            original_model_accuracy = loaded_model.evaluate(X_test_scaled, y_test)[1]
            st.session_state.original_model_accuracy = original_model_accuracy

            st.write(f"Original Model Accuracy: {original_model_accuracy}")

        if st.button("Apply Boundary Attack"):
            if hasattr(st.session_state, 'loaded_model'):
                loaded_model = st.session_state.loaded_model
                X_test_scaled = st.session_state.X_test_scaled
                y_test = st.session_state.y_test

            #Apply Boundary attack

            X_test_adv = boundary_attack_tabular(loaded_model, X_test_scaled, y_test)

            # Evaluate the perturbed model
            perturbed_model_accuracy = loaded_model.evaluate(X_test_adv, y_test)[1]
            #st.write(f"Perturbed Model Accuracy: {perturbed_model_accuracy}")

            # Display accuracy using a gauge visualization
        
            st.write(f"Original Model Accuracy:{round(original_model_accuracy, 2)}")
            st.progress(original_model_accuracy)

            st.write(f"Perturbed Model Accuracy: {round(perturbed_model_accuracy, 2)}")
            st.progress(perturbed_model_accuracy)

            st.title("Suggested Defenses")

            with st.container():
                st.subheader("Adversarial Training")
                st.write("Adversarial training is a machine learning technique that enhances model robustness. It involves exposing a model to adversarial examples, which are subtly modified inputs designed to mislead the model's predictions. By iteratively refining the model's response to such examples, adversarial training fortifies it against potential real-world attacks, making it more reliable and secure.")

            with st.container():
                st.subheader("Ensemble Models with Squeezed Features")
                st.write("Feature squeezing strengthens models against attacks, utilizing rescaling and bit reduction. While testing, three models were introduced: a multilayer perceptron, a wide and deep model, and a CNN churn prediction model. Ensemble models combine predictive power, fortifying the defense against adversarial threats.")



#If Carlini attack
    elif attack_option == "Carlini Attack":
        st.title("Carlini Attack")
        dataset_file = st.file_uploader("Upload Dataset", type=["csv"])

        if dataset_file:
            data = pd.read_csv(dataset_file)

            # Let the user select the target column
            target_column = st.selectbox("Select Target Column", data.columns)
            X = data.drop(columns=[target_column])
            y = data[target_column]

            # Performing the train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_train.shape, X_test.shape, y_train.shape, y_test.shape

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            smote=SMOTE(sampling_strategy='auto', random_state=23)
            X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

            under_sampler = RandomUnderSampler(sampling_strategy='auto', random_state=23)
            X_train_combined, y_train_combined = under_sampler.fit_resample(X_train_smote, y_train_smote)

            st.write(y_train_combined.value_counts())
        
        model_file = st.file_uploader("Upload .h5 Model", type=["h5"])

        if model_file:
            temp_model_location = "temp_model.h5"
            with open(temp_model_location, 'wb') as out:
                out.write(model_file.read())

            loaded_model = tf.keras.models.load_model(temp_model_location)
            st.session_state.loaded_model = loaded_model
            st.session_state.X_test_scaled = X_test_scaled
            st.session_state.y_test = y_test

            original_model_accuracy = loaded_model.evaluate(X_test_scaled, y_test)[1]
            st.session_state.original_model_accuracy = original_model_accuracy

            st.write(f"Original Model Accuracy: {original_model_accuracy}")

        if st.button("Apply Carlini Attack"):
            if hasattr(st.session_state, 'loaded_model'):
                loaded_model = st.session_state.loaded_model
                X_test_scaled = st.session_state.X_test_scaled
                y_test = st.session_state.y_test

                # Generate adversarial examples
                X_test_tensor = tf.convert_to_tensor(X_test_scaled, dtype=tf.float32)
                X_test_carlini = carlini_attack_binary(loaded_model, X_test_tensor, y_test)

                perturbed_accuracy = loaded_model.evaluate(X_test_carlini, y_test)[1]

                st.write(f"Original Model Accuracy:{round(original_model_accuracy, 2)}")
                st.progress(original_model_accuracy)

                st.write(f"Perturbed Model Accuracy: {round(perturbed_accuracy, 2)}")
                st.progress(perturbed_accuracy)

                st.title("Suggested Defenses")

            with st.container():
                st.subheader("Adversarial Training")
                st.write("Adversarial training is a machine learning technique that enhances model robustness. It involves exposing a model to adversarial examples, which are subtly modified inputs designed to mislead the model's predictions. By iteratively refining the model's response to such examples, adversarial training fortifies it against potential real-world attacks, making it more reliable and secure.")


#If B&B
    elif attack_option == "Brendel & Bethge Attack":
        st.title("Brendel & Bethge Attack")
        dataset_file = st.file_uploader("Upload Dataset", type=["csv"])

        if dataset_file:
            data = pd.read_csv(dataset_file)

            # Let the user select the target column
            target_column = st.selectbox("Select Target Column", data.columns)
            X = data.drop(columns=[target_column])
            y = data[target_column]

            # Performing the train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_train.shape, X_test.shape, y_train.shape, y_test.shape

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            smote=SMOTE(sampling_strategy='auto', random_state=23)
            X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

            under_sampler = RandomUnderSampler(sampling_strategy='auto', random_state=23)
            X_train_combined, y_train_combined = under_sampler.fit_resample(X_train_smote, y_train_smote)

            st.write(y_train_combined.value_counts())
        
        model_file = st.file_uploader("Upload .h5 Model", type=["h5"])

        if model_file:
            temp_model_location = "temp_model.h5"
            with open(temp_model_location, 'wb') as out:
                out.write(model_file.read())

            loaded_model = tf.keras.models.load_model(temp_model_location)
            st.session_state.loaded_model = loaded_model
            st.session_state.X_test_scaled = X_test_scaled
            st.session_state.y_test = y_test

            original_model_accuracy = loaded_model.evaluate(X_test_scaled, y_test)[1]
            st.session_state.original_model_accuracy = original_model_accuracy

            st.write(f"Original Model Accuracy: {original_model_accuracy}")

        if st.button("Apply Brendel & Bethge Attack"):
            if hasattr(st.session_state, 'loaded_model'):
                loaded_model = st.session_state.loaded_model
                X_test_scaled = st.session_state.X_test_scaled
                y_test = st.session_state.y_test

                X_test_brendel = brendel_bethge_attack(loaded_model, X_test_scaled, y_test)
                perturbed_accuracy = loaded_model.evaluate(X_test_brendel, y_test)[1]

                st.write(f"Original Model Accuracy:{round(original_model_accuracy, 2)}")
                st.progress(original_model_accuracy)

                st.write(f"Perturbed Model Accuracy: {round(perturbed_accuracy, 2)}")
                st.progress(perturbed_accuracy)

                st.title("Suggested Defenses")

            with st.container():
                st.subheader("Defensive Distillation")
                st.write("This approach enhances adversarial example crafting by leveraging insights from a teacher network. The teacher model initially trains on the original dataset, producing softened probabilities for more confident outputs. The student model learns from the teacher's expertise, aiming to fortify resilience against adversarial perturbations with nuanced insights.")



if __name__ == "__main__":
    main()