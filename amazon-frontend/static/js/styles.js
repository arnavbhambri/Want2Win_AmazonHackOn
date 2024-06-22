document.addEventListener("DOMContentLoaded", function() {
    // Login form validation
    const loginForm = document.getElementById('loginForm');
    loginForm.addEventListener('submit', function(event) {
        event.preventDefault();
        if (loginForm.checkValidity() === false) {
            event.stopPropagation();
        }
        loginForm.classList.add('was-validated');
    });

    // Registration form validation
    const registerForm = document.getElementById('registerForm');
    const password = document.getElementById('password');
    const confirmPassword = document.getElementById('confirm-password');

    registerForm.addEventListener('submit', function(event) {
        event.preventDefault();
        if (registerForm.checkValidity() === false) {
            event.stopPropagation();
        } else if (password.value !== confirmPassword.value) {
            confirmPassword.setCustomValidity("Passwords do not match");
        } else {
            confirmPassword.setCustomValidity("");
        }
        registerForm.classList.add('was-validated');
    });
});
