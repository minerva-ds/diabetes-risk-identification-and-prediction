let features = {
    'HighBP': false,  // bool
    'HighChol': false,  // bool
    'CholCheck': false,  // bool
    'BMI': 1,  // uint8
    'Smoker': false,  // bool
    'Stroke': false,  // bool
    'HeartDiseaseorAttack': false,  // bool
    'PhysActivity': false,  // bool
    'Fruits': false,  // bool
    'Veggies': false,  // bool
    'HvyAlcoholConsump': false,  // bool
    'AnyHealthcare': false,  // bool
    'NoDocbcCost': false,  // bool
    'GenHlth': 1,  // uint8
    'DiffWalk': false,  // bool
    'Sex': false,  // bool
    'Age': 1,  // uint8
    'MentHlth_binned': 1,  // uint8
    'PhysHlth_binned': 1,  // uint8
    'Education_binned': 1,  // uint8
    'Income_binned': 1,  // uint8
    'BMI_GenHlth_interaction': 0,  // uint8
    'Income_binned_GenHlth_interaction': 0,  // uint8
    'GenHlth_Age_interaction': 0,  // uint8
    'Age_PhysHlth_binned_interaction': 0  // uint8
};


function calculate_bmi() {
    let height_feet = parseFloat(document.querySelector("#height_feet").value);
    let height_inches = parseFloat(document.querySelector("#height_inches").value);
    let weight = parseFloat(document.querySelector("#weight").value);

    let bmi = 23; // Default BMI

    if (height_feet && (height_inches == 0 || height_inches) && weight) {
        // Convert height to meters
        let height_in_meters = (height_feet * 0.3048) + (height_inches * 0.0254);
        // Convert weight to kilograms
        let weight_in_kg = weight * 0.453592;

        // Calculate BMI
        bmi = weight_in_kg / (height_in_meters * height_in_meters);
        bmi = Math.round(bmi * 10) / 10; // Round to 1 decimal place
    } else {
        console.log('Using default BMI, could not calculate');
    }

    if (bmi < 12) bmi = 12;
    if (bmi > 60) bmi = 60;

    document.querySelector("#bmi_rating").innerHTML = bmi;
    return bmi;
}

function calculate_interaction_terms(features) {
    return {
        'BMI_GenHlth_interaction': features['BMI'] * features['GenHlth'],
        'Income_binned_GenHlth_interaction': features['Income_binned'] * features['GenHlth'],
        'GenHlth_Age_interaction': features['GenHlth'] * features['Age'],
        'Age_PhysHlth_binned_interaction': features['Age'] * features['PhysHlth_binned']
    };
}

function get_features() {
    let e = null;
    let fvalue = 0;

    for (let f in features) {
        //console.log('getting #'+f)
        e = document.querySelector("#" + f);

        if(!e){
            continue;
        }
        else if (e.type === 'checkbox') {
            fvalue = e.checked ? 1 : 0;
        } else if (e.type === 'number') {
            fvalue = parseFloat(e.value);
        } else if (e.tagName.toLowerCase() === 'select') {
            fvalue = parseInt(e.value);
        } else {
            console.log("skipping " + f + ". Could not identify input type");
            continue;
        }

        // Update feature only if the input is not empty or invalid
        if (!isNaN(fvalue) && fvalue !== null && fvalue !== undefined) {
            features[f] = fvalue;
        }
    }
    // Calculate BMI and update the BMI feature
    features['BMI'] = calculate_bmi();

    // Calculate interaction terms and update the features object
    let interaction_terms = calculate_interaction_terms(features);
    for (let key in interaction_terms) {
        features[key] = interaction_terms[key];
    }
    console.log("Features:", features);

    // Return only the feature values, maintaining order
    return Object.values(features);
}

async function run_model() {
    try {
        let feature_values = get_features();
        console.log("Feature Values:", feature_values);

        // Load the ONNX model
        let session = await ort.InferenceSession.create('models/catboost_model_no_zipmap.onnx');

        // Create tensor from the features
        let input_tensor = new ort.Tensor('float32', new Float32Array(feature_values), [1, feature_values.length]);

        // Run the model
        let results = await session.run({ 'input': input_tensor });

        // Get the probability of class 1 (diabetes risk)
        let probability_class_1 = results.probabilities.data[1];
        let threshold = 0.36;

        // Rescale the probability so that 0.28 corresponds to 50%
        let adjusted_probability = (probability_class_1 - threshold) / (2 * (1 - threshold)) + 0.5;
        let adjusted_percentage = (adjusted_probability * 100).toFixed(2);

        // Determine the label based on the original threshold
        let label = probability_class_1 >= threshold ? 1 : 0;

        // Set the result message
        let result_ele = document.querySelector("#result");
        result_ele.innerHTML = `Preliminary assessment suggests a ${adjusted_percentage}% likelihood of diabetes risk.`;

        // Add class and additional message based on the label
        if (!label) {
            result_ele.className = 'ok';
        } else {
            result_ele.className = 'not_ok';
            result_ele.innerHTML += "<br><br>This prescreening tool suggests you may be at risk. It would be prudent to schedule an appointment with a medical professional for further evaluation.";
        }

    } catch (err) {
        console.error('Error during inference:', err);
    }
}

document.addEventListener("DOMContentLoaded", function() {
    document.querySelector("#predict_button").addEventListener("click", run_model);

    let heightFeetElement = document.querySelector("#height_feet");
    let heightInchesElement = document.querySelector("#height_inches");
    let weightElement = document.querySelector("#weight");
    heightFeetElement.addEventListener("input", calculate_bmi);
    heightInchesElement.addEventListener("input", calculate_bmi);
    weightElement.addEventListener("input", calculate_bmi);
    calculate_bmi()
});