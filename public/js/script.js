let loader = document.getElementById("loader");
if(loader){
    loader.addEventListener("click", ()=>{
        let spinner = document.getElementById("spinner");
        let classifierTrain = document.getElementById("classifier_train");
        let classifierTest = document.getElementById("classifier_test");
        if(spinner || classifierTrain || classifierTest){
            spinner.style.display="block";
            loader.style.display="none";
            if(classifierTrain)
                classifierTrain.innerHTML= "Your classifier is training. Please wait for a while.";
            if(classifierTest)
                classifierTest.innerHTML= "Your image is testing. Please wait for a while."

        }
    })
}