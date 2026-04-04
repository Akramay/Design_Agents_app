const express = require('express');
const path = require('path');
const app = express();

app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));
app.use(express.static(path.join(__dirname, 'public')));


app.get('/', (req, res) => {
    res.render('pages/home'); 
});

// Start the server
const PORT = 3000;
app.listen(PORT, () => {
    console.log(`Server is running! Open http://localhost:${PORT} in your browser.`);
});