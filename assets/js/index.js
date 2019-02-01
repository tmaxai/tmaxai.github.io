// turn img alt into caption

$('.post > p > img[alt]').replaceWith(function () {
    return '<figure>'
        + '<a href="' + $(this).attr('src') + '" class="mg-link">'
        + '<img src="' + $(this).attr('src') + '"/></a>'
        //+ '<figcaption class="caption">' + $(this).attr('alt') + '</figcaption>'
        + '</figure>';
});

$('.about > p > img[alt]').replaceWith(function () {
    return '<figure>'
        + '<a href="' + $(this).attr('src') + '" class="mg-link">'
        + '<img src="' + $(this).attr('src') + '"/></a>'
        //+ '<figcaption class="caption">' + $(this).attr('alt') + '</figcaption>'
        + '</figure>';
});

// and connect magnific popup image viewer
$('.mg-link').magnificPopup({
    type: 'image',
    closeOnContentClick: true
});
