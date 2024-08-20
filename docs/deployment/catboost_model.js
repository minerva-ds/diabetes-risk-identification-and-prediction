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
    'AnyHealthcare': false,  // bool
    'NoDocbcCost': false,  // bool
    'GenHlth': 1,  // uint8
    'DiffWalk': false,  // bool
    'Sex': false,  // bool
    'Age': 1,  // uint8
    'MentHlth': 0,  // uint8 (0-30 days, adjusted from MentHlth_binned)
    'PhysHlth': 0,  // uint8 (0-30 days, adjusted from PhysHlth_binned)
    'Education': 1,  // uint8 (adjusted from Education_binned)
    'Income': 1,  // uint8 (adjusted from Income_binned)
    'BMI_Age_interaction': 0,  // uint8 (interaction term calculated dynamically)
    'Income_GenHlth_interaction': 0,  // uint8 (interaction term calculated dynamically)
    'PhysHlth_BMI_interaction': 0,  // uint8 (interaction term calculated dynamically)
    'MentHlth_BMI_interaction': 0  // uint8 (interaction term calculated dynamically)
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
        'BMI_Age_interaction': features['BMI'] * features['Age'],
        'Income_GenHlth_interaction': features['Income'] * features['GenHlth'],
        'PhysHlth_BMI_interaction': features['PhysHlth'] * features['BMI'],
        'MentHlth_BMI_interaction': features['MentHlth'] * features['BMI']
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
        let threshold = 0.32;

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
function predict() {
    document.querySelector("#predict_button").addEventListener("click", async function() {
        await run_model(); // Wait for the async function to complete
        scroll_to_prediction_box(); // Scroll to the bottom of the #prediction_box after run_model completes
    });
}

function bmi_listeners() {
    let height_feet_element = document.querySelector("#height_feet");
    let height_inches_element = document.querySelector("#height_inches");
    let weight_element = document.querySelector("#weight");

    height_feet_element.addEventListener("input", calculate_bmi);
    height_inches_element.addEventListener("input", calculate_bmi);
    weight_element.addEventListener("input", calculate_bmi);
}

function advanced_view_toggle() {
    let advanced_view_toggle = document.querySelector("#advancedViewToggle");
    advanced_view_toggle.addEventListener("change", toggle_advanced_view);

    // Trigger the initial state of the advanced features based on the checkbox status
    advanced_view_toggle.dispatchEvent(new Event("change"));
}

function toggle_advanced_view() {
    let advanced_view_toggle = document.querySelector("#advancedViewToggle");
    let advanced_features = document.querySelectorAll(".advanced-feature");

    if (advanced_view_toggle.checked) {
        advanced_features.forEach(function(feature) {
            feature.style.display = "block"; // Show advanced features
        });
    } else {
        advanced_features.forEach(function(feature) {
            feature.style.display = "none"; // Hide advanced features
        });
    }
}

function advanced_view_modal() {
    let modal = document.getElementById("advancedViewDialog");
    let info_icon = document.getElementById("advancedViewInfo");
    let close_modal = document.getElementById("closeModal");

    info_icon.addEventListener("click", function() {
        modal.style.display = "block";
    });

    close_modal.addEventListener("click", function() {
        modal.style.display = "none";
    });

    window.addEventListener("click", function(event) {
        if (event.target == modal) {
            modal.style.display = "none";
        }
    });
}
function scroll_to_prediction_box() {
    let predict_box = document.querySelector("#prediction_box");
    if (predict_box) {
        predict_box.scrollIntoView({ behavior: "smooth", block: "end" });
    }
}


document.addEventListener("DOMContentLoaded", function() {
    predict();
    bmi_listeners();
    calculate_bmi();  // Calculate BMI on page load
    advanced_view_toggle();
    advanced_view_modal();
});