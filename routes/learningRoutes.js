const express = require('express');
const router = express.Router();

// /learn
router.get('/', (req, res) => {
    res.render('pages/learning');
});

// /learn/questions
router.get('/questions', (req, res) => {
    res.render('pages/questions');
});

// /learn/video
router.get('/video', (req, res) => {
    res.render('pages/video');
});

// /learn/explain
router.get('/explain', (req, res) => {
    res.render('pages/explain');
});

module.exports = router;