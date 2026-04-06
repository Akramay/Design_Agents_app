const express = require('express');
const path = require('path');
const app = express();

// View engine
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));

// Static files
app.use(express.static(path.join(__dirname, 'public')));

// Import routes
const learningRoutes = require('./routes/learningRoutes');

// Routes
app.get('/', (req, res) => {
    res.render('pages/home');
});

// Use learning routes
app.use('/learn', learningRoutes);

// Start server
const PORT = 3000;
app.listen(PORT, () => {
    console.log(`Server is running! Open http://localhost:${PORT}`);
});