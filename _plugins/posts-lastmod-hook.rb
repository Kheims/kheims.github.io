#!/usr/bin/env ruby
#
# Check for changed posts

Jekyll::Hooks.register :posts, :post_init do |post|

  commit_num = `git rev-list --count HEAD "#{ post.path }"`

  if commit_num.to_i > 1
    lastmod_date = `git log -1 --pretty="%ad" --date=iso "#{ post.path }"`
    post.data['last_modified_at'] = lastmod_date
  end

end

Jekyll::Hooks.register :posts, :pre_render do |post|
  next if post.data['read_time']

  word_count = post.content.gsub(/```.*?```/m, " ").scan(/[[:word:]]+/).size
  minutes = [(word_count / 220.0).ceil, 1].max

  post.data['word_count'] = word_count
  post.data['read_time'] = "#{minutes} min read"
end
