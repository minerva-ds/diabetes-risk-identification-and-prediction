document.addEventListener("DOMContentLoaded", function() {
    document.querySelector("#predict_button").addEventListener("click", run_model);
});

let features = {
    'HighBP': 0,
    'HighChol': 0,
    'CholCheck': 0,
    'BMI': 1,
    'Smoker': 0,
    'Stroke': 0,
    'HeartDiseaseorAttack': 0,
    'PhysActivity': 0,
    'Fruits': 0,
    'Veggies': 0,
    'HvyAlcoholConsump': 0,
    'AnyHealthcare': 0,
    'NoDocbcCost': 0,
    'GenHlth': 1,
    'DiffWalk': 0,
    'Sex': 0,
    'Age': 1,
    'MentHlth_binned': 1,
    'PhysHlth_binned': 1,
    'Education_binned': 1,
    'Income_binned': 1,
    'PhysHlth_binned_DiffWalk_interaction': 0,
    'BMI_Age_interaction': 0,
    'PhysActivity_BMI_interaction': 0,
    'HighChol_HighBP_interaction': 0,
    'MentHlth_binned_DiffWalk_interaction': 0,
    'Income_binned_Education_binned_interaction': 0,
    'Fruits_Veggies_interaction': 0,
    'Smoker_HvyAlcoholConsump_interaction': 0,
    'GenHlth_AnyHealthcare_interaction': 0
};

function calculate_bmi() {
    let heightFeetElement = document.querySelector("#height_feet");
    let heightInchesElement = document.querySelector("#height_inches");
    let weightElement = document.querySelector("#weight");

    let height_feet = heightFeetElement ? parseInt(heightFeetElement.value) : null;
    let height_inches = heightInchesElement ? parseInt(heightInchesElement.value) : null;
    let weight = weightElement ? parseFloat(weightElement.value) : null;

    if (height_feet && height_inches && weight) {
        let height_in_meters = (height_feet * 0.3048) + (height_inches * 0.0254);
        return Math.round(weight / (height_in_meters * height_in_meters));
    } else {
        return 23;
    }
}

function calculate_interaction_terms(features) {
    return {
        'PhysHlth_binned_DiffWalk_interaction': features['PhysHlth_binned'] * features['DiffWalk'],
        'BMI_Age_interaction': features['BMI'] * features['Age'],
        'PhysActivity_BMI_interaction': features['PhysActivity'] * features['BMI'],
        'HighChol_HighBP_interaction': features['HighChol'] * features['HighBP'],
        'MentHlth_binned_DiffWalk_interaction': features['MentHlth_binned'] * features['DiffWalk'],
        'Income_binned_Education_binned_interaction': features['Income_binned'] * features['Education_binned'],
        'Fruits_Veggies_interaction': features['Fruits'] * features['Veggies'],
        'Smoker_HvyAlcoholConsump_interaction': features['Smoker'] * features['HvyAlcoholConsump'],
        'GenHlth_AnyHealthcare_interaction': features['GenHlth'] * features['AnyHealthcare']
    };
}

function get_features() {
    let e = null;
    let fvalue = 0;

    for (let f in features) {
        console.log('getting #'+f)
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

    // Return only the feature values, maintaining order
    return Object.values(features);
}

// Example of how to use get_features with your model prediction
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

        // Log and display the results
        console.log('Model Results:', results);
        let label = results.label.data[0];
        console.log('results',results)

        let probabilities = results.probabilities.cpuData;
        document.querySelector("#result").innerHTML = `Predicted Label: ${label}<br>Probabilities: ${probabilities.join(', ')}`;


    } catch (err) {
        console.error('Error during inference:', err);
    }
}
