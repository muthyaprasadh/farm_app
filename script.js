function upload() {
    let file = document.getElementById("image").files[0];

    let formData = new FormData();
    formData.append("image", file);

    fetch("http://farm-alb-1952832472.ap-south-1.elb.amazonaws.com/predict", {
        method: "POST",
        body: formData
    })
    .then(res => res.json())
    .then(data => {
        document.getElementById("result").innerText =
            "Disease: " + data.disease;
    });
}