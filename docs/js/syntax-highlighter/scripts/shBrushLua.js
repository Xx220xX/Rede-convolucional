/**
 * SyntaxHighlighter
 * http://alexgorbatchev.com/SyntaxHighlighter
 *
 * SyntaxHighlighter is donationware. If you are using it, please donate.
 * http://alexgorbatchev.com/SyntaxHighlighter/donate.html
 *
 * @version
 * 3.0.83 (July 02 2010)
 *
 * @copyright
 * Copyright (C) 2004-2010 Alex Gorbatchev.
 *
 * @license
 * Dual licensed under the MIT and GPL licenses.
 */
;(function()
{
    // CommonJS
    typeof(require) != 'undefined' ? SyntaxHighlighter = require('shCore').SyntaxHighlighter : null;

    function Brush()
    {
        // Contributed by Gheorghe Milas and Ahmad Sherif

        var keywords =  'require do break end elseif else this function if local nil or not return repeat and until then while';

        var funcs = 'math\\.\\w+ string\\.\\w+ os\\.\\w+ debug\\.\\w+ io\\.\\w+ error fopen dofile coroutine\\.\\w+ arg ipairs getmetatable loadfile longjmp loadstring rawget print seek rawset assert setmetatable tostring tonumber loadlib';

        var special =  'true false __sub  __mul  __div  __mod  __pow  __unm  __concat  __len  __eq  __lt  __le  __call  __gc  __index  __newindex';

        this.regexList = [
            // { regex: SyntaxHighlighter.regexLib.singleLinePerlComments, css: 'comments' },
            { regex: /^\s*@\w+/gm, 										css: 'decorator' },
            { regex: /--\[(=*)\[(.|\n)*?\]\1\]/gm, 						css: 'comments' },
            { regex: /--(=*)(.)*/gm, 						css: 'comments' },
            { regex: /"(?!")(?:\.|\\\"|[^\""\n])*"/gm, 					css: 'string' },
            { regex: /'(?!')(?:\.|(\\\')|[^\''\n])*'/gm, 				css: 'string' },
            { regex: /\+|\-|\*|\/|\%|=|==/gm, 							css: 'keyword' },
            { regex: /\b\d+\.?\w*/g, 									css: 'value' },
            { regex: /\.\./gm, 									css: 'color2' },
            { regex: /;/gm, 									css: 'color2' },
            { regex: /\./g, 									css: 'functions' },
            { regex: /\(/g, 									css: 'color2' },
            { regex: /\)/g, 									css: 'color2' },
            { regex: new RegExp(this.getKeywords(funcs), 'gmi'),		css: 'functions' },
            { regex: new RegExp(this.getKeywords(keywords), 'gm'), 		css: 'keyword' },
            { regex: new RegExp(this.getKeywords(special), 'gm'), 		css: 'color1' },
            { regex: /./g, 									css: 'color3' }
        ];

        this.forHtmlScript(SyntaxHighlighter.regexLib.aspScriptTags);
    };

    Brush.prototype	= new SyntaxHighlighter.Highlighter();
    Brush.aliases	= ['lua', 'Lua'];

    SyntaxHighlighter.brushes.Lua = Brush;

    // CommonJS
    typeof(exports) != 'undefined' ? exports.Brush = Brush : null;
})();
