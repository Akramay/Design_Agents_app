const express = require('express');
const path = require('path');
const app = express();

// View engine
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));

// Static files
app.use(express.static(path.join(__dirname, 'public')));
app.use(express.json({ limit: '50mb' }));

// Request tracing middleware
app.use((req, res, next) => {
    const start = process.hrtime.bigint();
    const now = new Date().toISOString();
    console.log(`[BACKEND] ${now} --> ${req.method} ${req.originalUrl}`);

    res.on('finish', () => {
        const durationMs = Number(process.hrtime.bigint() - start) / 1e6;
        console.log(
            `[BACKEND] ${now} <-- ${req.method} ${req.originalUrl} ${res.statusCode} ${durationMs.toFixed(1)}ms`
        );
    });

    next();
});

// Import routes
const learningRoutes = require('./routes/learningRoutes');
const tutorApiRoutes = require('./routes/tutorApi');
app.use('/api/tutor', tutorApiRoutes);
// Routes
app.get('/', (req, res) => {
    res.render('pages/home');
});

// Use learning routes
// Use learning routes
app.use('/learn', learningRoutes);
app.use('/learn/api', tutorApiRoutes);
// Start server
const PORT = 3000;
app.listen(PORT, () => {
    console.log(`Server is running! Open http://localhost:${PORT}`);
});