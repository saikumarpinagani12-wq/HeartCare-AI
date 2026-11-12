
const ecgFile = document.getElementById("ecgFile");
const fileName = document.getElementById("fileName");

if (ecgFile && fileName) {
    ecgFile.addEventListener("change", function () {
        if (this.files.length > 0) {
            fileName.textContent =
                "Selected: " + this.files[0].name;
        } else {
            fileName.textContent = "No file selected";
        }
    });
}
